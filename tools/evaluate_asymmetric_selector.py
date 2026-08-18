#!/usr/bin/env python3
"""Audit the exact asymmetric SAM selector against DOTA GT OBBs.

Ground truth is used only after candidate generation and selection. Sparse
images run all three MobileSAM views; dense images run the original single
radius watershed path. Results are reported separately for both branches.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmengine.config import Config
from mmrotate.models.losses.utils import filter_masks
from projects.point2rbox_v3_multiscale.multiscale_voronoi_loss import (
    MultiScaleVoronoiWatershedLoss,
)
from tools.evaluate_candidate_oracle import (
    choose_samples, load_annotation, locate_image, rotated_iou,
    training_image_views, watershed_candidates,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=(
        'configs/experiments/point2rbox_v3_a_ms_hard_dev.py'))
    parser.add_argument('--data-root', default='data/split_ss_dota/trainval')
    parser.add_argument('--sam-checkpoint', default='mobile_sam.pt')
    parser.add_argument('--samples', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', default=(
        'work_dirs/asymmetric_selector_audit_500_seed0.json'))
    parser.add_argument('--progress-every', type=int, default=10)
    return parser.parse_args()


def candidate_id(candidate):
    return {
        'view': candidate.view_type,
        'scale': candidate.source_scale,
        'multimask_index': candidate.source_index,
    }


def find_baseline(loss_module, candidates, instance_index, points, labels,
                  image_tensor, image_np):
    originals = [item for item in candidates if item.view_type == 'original']
    if not originals:
        return None
    masks = [item.mask for item in originals]
    scores = np.asarray([item.score for item in originals])
    index, _, _ = filter_masks(
        image_tensor, masks, scores, int(labels[instance_index]), image_np,
        points[instance_index], loss_module.mask_filter_config, False)
    return originals[index]


def empty_summary():
    return {
        'images': 0,
        'instances': 0,
        'baseline_iou_sum': 0.0,
        'oracle_iou_sum': 0.0,
        'selected_iou_sum': 0.0,
        'replacements': 0,
        'wrong_replacements': 0,
        'correct_replacements': 0,
        'ties': 0,
        'fallbacks': 0,
        'oracle_views': defaultdict(int),
        'selected_views': defaultdict(int),
        'per_class': defaultdict(empty_summary),
    }


def update_summary(summary, record):
    summary['instances'] += 1
    for key in ('baseline_iou', 'oracle_iou', 'selected_iou'):
        summary[f'{key}_sum'] += record[key]
    summary['replacements'] += int(record['replaced'])
    summary['wrong_replacements'] += int(record['wrong_replacement'])
    summary['correct_replacements'] += int(record['correct_replacement'])
    summary['ties'] += int(record['replacement_tie'])
    summary['fallbacks'] += int(not record['replaced'])
    summary['oracle_views'][record['oracle']['view']] += 1
    summary['selected_views'][record['selected']['view']] += 1


def finalize(summary):
    result = dict(summary)
    per_class = result.pop('per_class')
    count = result['instances']
    for key in ('baseline_iou', 'oracle_iou', 'selected_iou'):
        result[key] = result.pop(f'{key}_sum') / count if count else 0.0
    result['oracle_gain'] = result['oracle_iou'] - result['baseline_iou']
    result['selected_gain'] = result['selected_iou'] - result['baseline_iou']
    result['selector_regret'] = result['oracle_iou'] - result['selected_iou']
    result['replacement_rate'] = (
        result['replacements'] / count if count else 0.0)
    result['wrong_replacement_rate_all'] = (
        result['wrong_replacements'] / count if count else 0.0)
    replacements = result['replacements']
    result['wrong_replacement_rate_replaced'] = (
        result['wrong_replacements'] / replacements if replacements else 0.0)
    result['fallback_rate'] = result['fallbacks'] / count if count else 0.0
    result['oracle_views'] = dict(result['oracle_views'])
    result['selected_views'] = dict(result['selected_views'])
    result['per_class'] = {
        name: finalize(values) for name, values in per_class.items()
    }
    return result


def make_record(obj, candidates, baseline, selected):
    ious = [rotated_iou(item.mask, obj['polygon']) for item in candidates]
    oracle_index = int(np.argmax(ious)) if ious else None
    oracle = candidates[oracle_index] if oracle_index is not None else None
    baseline_iou = (rotated_iou(baseline.mask, obj['polygon'])
                    if baseline is not None else 0.0)
    selected_iou = (rotated_iou(selected.mask, obj['polygon'])
                    if selected is not None else 0.0)
    oracle_iou = ious[oracle_index] if oracle_index is not None else 0.0
    replaced = selected is not None and selected is not baseline
    epsilon = 1e-12
    return {
        'class_name': obj['class_name'],
        'difficulty': obj['difficulty'],
        'candidate_count': len(candidates),
        'baseline_iou': baseline_iou,
        'oracle_iou': oracle_iou,
        'selected_iou': selected_iou,
        'oracle_gain': oracle_iou - baseline_iou,
        'selected_gain': selected_iou - baseline_iou,
        'selector_regret': oracle_iou - selected_iou,
        'replaced': replaced,
        'wrong_replacement': replaced and selected_iou < baseline_iou - epsilon,
        'correct_replacement': replaced and selected_iou > baseline_iou + epsilon,
        'replacement_tie': replaced and abs(selected_iou - baseline_iou) <= epsilon,
        'baseline': candidate_id(baseline) if baseline is not None else None,
        'oracle': candidate_id(oracle) if oracle is not None else {
            'view': 'missing', 'scale': None, 'multimask_index': None},
        'selected': candidate_id(selected) if selected is not None else {
            'view': 'missing', 'scale': None, 'multimask_index': None},
        'candidates': [dict(candidate_id(item), iou=iou,
                            sam_score=item.score,
                            boundary_contact_ratio=item.boundary_contact_ratio)
                       for item, iou in zip(candidates, ious)],
    }


def build_loss_and_predictor(args):
    cfg = Config.fromfile(args.config)
    loss_cfg = dict(cfg.model.bbox_head.loss_voronoi)
    loss_cfg.pop('type', None)
    loss_cfg['sam_checkpoint'] = args.sam_checkpoint
    loss_cfg.setdefault('loss_weight', 5.0)
    loss_cfg.setdefault('mask_filter_config', cfg.mask_filter_config)
    loss_cfg.setdefault('sam_instance_thr', cfg.sam_instance_thr)
    loss_cfg.setdefault('sam_sample_rules', cfg.sam_sample_rules)
    loss_module = MultiScaleVoronoiWatershedLoss(**loss_cfg)
    predictor = loss_module._build_predictor(args.device)
    return loss_module, predictor


def main():
    args = parse_args()
    import torch

    data_root = Path(args.data_root)
    annotations = choose_samples(
        data_root / 'annfiles', args.samples, args.seed)
    loss_module, predictor = build_loss_and_predictor(args)
    summaries = {'sam_sparse': empty_summary(),
                 'watershed_dense': empty_summary()}
    image_records = []
    started = time.perf_counter()

    for image_index, annotation_path in enumerate(annotations, 1):
        objects = load_annotation(annotation_path)
        image_path = locate_image(data_root / 'images', annotation_path.stem)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        normalized, image_np = training_image_views(bgr)
        image_tensor = torch.from_numpy(normalized).permute(2, 0, 1)
        points = np.asarray([obj['center'] for obj in objects], np.float32)
        labels = np.asarray([obj['class_id'] for obj in objects], np.int64)
        records = []

        if len(objects) <= loss_module.sam_instance_thr:
            branch = 'sam_sparse'
            by_instance = loss_module.candidate_pool.generate(
                predictor, image_np, points, labels,
                sample_rules=loss_module.sam_sample_rules)
            for instance_index, (obj, candidates) in enumerate(
                    zip(objects, by_instance)):
                baseline = find_baseline(
                    loss_module, candidates, instance_index, points, labels,
                    image_tensor, image_np)
                selected = loss_module._select_asymmetric_candidate(
                    instance_index, candidates, points, labels, image_tensor,
                    image_np) if candidates else None
                records.append(make_record(
                    obj, candidates, baseline, selected))
        else:
            branch = 'watershed_dense'
            masks_by_instance, _ = watershed_candidates(
                normalized, points, labels, (1.0,))
            for obj, masks in zip(objects, masks_by_instance):
                # A single radius has one effective candidate; the dense
                # branch is unchanged, so baseline/oracle/selected coincide.
                from projects.point2rbox_v3_multiscale.candidate_pool import Candidate
                candidates = [Candidate(mask, 1.0, 1.0, index, (1.0,),
                                        view_type='watershed')
                              for index, mask in enumerate(masks)]
                chosen = candidates[0] if candidates else None
                records.append(make_record(obj, candidates, chosen, chosen))

        summaries[branch]['images'] += 1
        for record in records:
            update_summary(summaries[branch], record)
            update_summary(
                summaries[branch]['per_class'][record['class_name']], record)
        image_records.append({
            'image_id': annotation_path.stem,
            'branch': branch,
            'instance_count': len(objects),
            'objects': records,
        })
        if image_index % args.progress_every == 0 or image_index == len(annotations):
            print(json.dumps({
                'done': image_index,
                'total': len(annotations),
                'sam_sparse_images': summaries['sam_sparse']['images'],
                'watershed_dense_images': summaries['watershed_dense']['images'],
                'elapsed_seconds': round(time.perf_counter() - started, 1),
            }), flush=True)

    output = {
        'protocol': {
            'samples': len(annotations),
            'seed': args.seed,
            'sample_ids': [path.stem for path in annotations],
            'sam_instance_thr': loss_module.sam_instance_thr,
            'config': args.config,
            'selector': loss_module.selector,
            'sam_views': ['texture', 'original', 'zoom'],
            'dense_watershed_radii': [1.0],
        },
        'summary': {name: finalize(values)
                    for name, values in summaries.items()},
        'images': image_records,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    concise = {
        name: {key: values[key] for key in (
            'images', 'instances', 'baseline_iou', 'oracle_iou',
            'selected_iou', 'oracle_gain', 'selected_gain',
            'selector_regret', 'replacement_rate',
            'wrong_replacement_rate_all',
            'wrong_replacement_rate_replaced')}
        for name, values in output['summary'].items()
    }
    print(json.dumps(concise, indent=2), flush=True)
    print(f'Wrote {output_path}', flush=True)


if __name__ == '__main__':
    main()
