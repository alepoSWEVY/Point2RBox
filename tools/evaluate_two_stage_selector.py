#!/usr/bin/env python3
"""Evaluate baseline-guided two-stage multi-view SAM selectors on DOTA."""

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

from mmrotate.models.losses.utils import filter_masks
from tools.evaluate_asymmetric_selector import build_loss_and_predictor
from tools.evaluate_candidate_oracle import (
    choose_samples, load_annotation, locate_image, rotated_iou,
    training_image_views, watershed_candidates,
)


STRATEGIES = {
    'baseline_original': ('baseline', 0.0),
    'cross_m0': ('cross', 0.0),
    'reliability_m0': ('reliability', 0.0),
    'reliability_m005': ('reliability', 0.05),
    'reliability_m010': ('reliability', 0.10),
}


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
        'work_dirs/two_stage_selector_audit_500_seed0.json'))
    parser.add_argument('--progress-every', type=int, default=25)
    return parser.parse_args()


def mask_iou(left, right):
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 0.0


def center_score(candidate, point):
    contours, _ = cv2.findContours(
        candidate.mask.astype(np.uint8), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    (cx, cy), (width, height), _ = cv2.minAreaRect(contour)
    if width <= 0 or height <= 0:
        return 0.0
    distance = np.linalg.norm(np.asarray((cx, cy)) - point)
    return float(np.exp(-distance / np.sqrt(width * height)))


def select_one_per_view(loss_module, candidates, instance_index, points,
                        labels, image_tensor, image_np):
    selected = {}
    for view in ('texture', 'original', 'zoom'):
        items = [item for item in candidates if item.view_type == view]
        if not items:
            continue
        index, _, _ = filter_masks(
            image_tensor, [item.mask for item in items],
            np.asarray([item.score for item in items]),
            int(labels[instance_index]), image_np, points[instance_index],
            loss_module.mask_filter_config, False)
        selected[view] = items[index]
    return selected


def candidate_scores(view_candidates, point):
    result = {}
    for view, candidate in view_candidates.items():
        others = [item for other_view, item in view_candidates.items()
                  if other_view != view]
        cross = (float(np.mean([
            mask_iou(candidate.mask, other.mask) for other in others
        ])) if others else 0.0)
        sam = float(np.clip(candidate.score, 0.0, 1.0))
        center = center_score(candidate, point)
        result[view] = {
            'cross': cross,
            'sam': sam,
            'center': center,
            'reliability': 0.5 * cross + 0.3 * sam + 0.2 * center,
        }
    return result


def choose(view_candidates, scores, mode, margin):
    baseline = view_candidates.get('original')
    if baseline is None or mode == 'baseline':
        return baseline
    alternatives = [view for view in ('texture', 'zoom')
                    if view in view_candidates]
    if not alternatives:
        return baseline
    score_key = 'cross' if mode == 'cross' else 'reliability'
    best_view = max(alternatives, key=lambda view: scores[view][score_key])
    if scores[best_view][score_key] > scores['original'][score_key] + margin:
        return view_candidates[best_view]
    return baseline


def empty_stats():
    return {
        'instances': 0, 'selected_iou_sum': 0.0, 'replacements': 0,
        'wrong': 0, 'correct': 0, 'ties': 0, 'views': defaultdict(int),
        'per_class': defaultdict(empty_stats),
    }


def update_stats(stats, record):
    stats['instances'] += 1
    stats['selected_iou_sum'] += record['selected_iou']
    stats['replacements'] += int(record['replaced'])
    stats['wrong'] += int(record['wrong'])
    stats['correct'] += int(record['correct'])
    stats['ties'] += int(record['tie'])
    stats['views'][record['selected_view']] += 1


def finalize_stats(stats, baseline_iou):
    result = dict(stats)
    per_class = result.pop('per_class')
    count = result['instances']
    result['selected_iou'] = (
        result.pop('selected_iou_sum') / count if count else 0.0)
    result['selected_gain'] = result['selected_iou'] - baseline_iou
    result['replacement_rate'] = (
        result['replacements'] / count if count else 0.0)
    result['wrong_replacement_rate_all'] = (
        result['wrong'] / count if count else 0.0)
    result['wrong_replacement_rate_replaced'] = (
        result['wrong'] / result['replacements']
        if result['replacements'] else 0.0)
    result['views'] = dict(result['views'])
    result['per_class'] = {
        name: finalize_stats(values, 0.0) for name, values in per_class.items()
    }
    return result


def identity(candidate):
    return {
        'view': candidate.view_type,
        'scale': candidate.source_scale,
        'multimask_index': candidate.source_index,
    }


def main():
    args = parse_args()
    import torch

    data_root = Path(args.data_root)
    annotations = choose_samples(
        data_root / 'annfiles', args.samples, args.seed)
    loss_module, predictor = build_loss_and_predictor(args)
    strategy_stats = {name: empty_stats() for name in STRATEGIES}
    sparse = {
        'images': 0, 'instances': 0, 'baseline_sum': 0.0,
        'raw_oracle_sum': 0.0, 'stage1_oracle_sum': 0.0,
        'raw_oracle_views': defaultdict(int),
        'stage1_oracle_views': defaultdict(int),
        'per_class_baseline': defaultdict(lambda: [0, 0.0]),
    }
    dense = {'images': 0, 'instances': 0, 'iou_sum': 0.0,
             'per_class': defaultdict(lambda: [0, 0.0])}
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
        object_records = []

        if len(objects) <= loss_module.sam_instance_thr:
            branch = 'sam_sparse'
            sparse['images'] += 1
            by_instance = loss_module.candidate_pool.generate(
                predictor, image_np, points, labels,
                sample_rules=loss_module.sam_sample_rules)
            for instance_index, (obj, raw_candidates) in enumerate(
                    zip(objects, by_instance)):
                view_candidates = select_one_per_view(
                    loss_module, raw_candidates, instance_index, points,
                    labels, image_tensor, image_np)
                if 'original' not in view_candidates:
                    continue
                scores = candidate_scores(
                    view_candidates, points[instance_index])
                baseline = view_candidates['original']
                baseline_iou = rotated_iou(baseline.mask, obj['polygon'])
                raw_ious = [rotated_iou(item.mask, obj['polygon'])
                            for item in raw_candidates]
                raw_index = int(np.argmax(raw_ious))
                raw_oracle = raw_candidates[raw_index]
                stage1_items = list(view_candidates.values())
                stage1_ious = [rotated_iou(item.mask, obj['polygon'])
                               for item in stage1_items]
                stage1_index = int(np.argmax(stage1_ious))
                stage1_oracle = stage1_items[stage1_index]
                sparse['instances'] += 1
                sparse['baseline_sum'] += baseline_iou
                sparse['raw_oracle_sum'] += raw_ious[raw_index]
                sparse['stage1_oracle_sum'] += stage1_ious[stage1_index]
                sparse['per_class_baseline'][obj['class_name']][0] += 1
                sparse['per_class_baseline'][obj['class_name']][1] += baseline_iou
                sparse['raw_oracle_views'][raw_oracle.view_type] += 1
                sparse['stage1_oracle_views'][stage1_oracle.view_type] += 1
                selections = {}
                epsilon = 1e-12
                for name, (mode, margin) in STRATEGIES.items():
                    selected = choose(view_candidates, scores, mode, margin)
                    selected_iou = rotated_iou(selected.mask, obj['polygon'])
                    replaced = selected is not baseline
                    record = {
                        'selected_iou': selected_iou,
                        'selected_view': selected.view_type,
                        'replaced': replaced,
                        'wrong': replaced and selected_iou < baseline_iou - epsilon,
                        'correct': replaced and selected_iou > baseline_iou + epsilon,
                        'tie': replaced and abs(selected_iou - baseline_iou) <= epsilon,
                    }
                    update_stats(strategy_stats[name], record)
                    update_stats(
                        strategy_stats[name]['per_class'][obj['class_name']],
                        record)
                    selections[name] = dict(record, candidate=identity(selected))
                object_records.append({
                    'class_name': obj['class_name'],
                    'baseline_iou': baseline_iou,
                    'raw9_oracle_iou': raw_ious[raw_index],
                    'stage1_oracle_iou': stage1_ious[stage1_index],
                    'raw9_oracle': identity(raw_oracle),
                    'stage1_oracle': identity(stage1_oracle),
                    'view_candidates': {
                        view: dict(identity(item), iou=rotated_iou(
                            item.mask, obj['polygon']), **scores[view])
                        for view, item in view_candidates.items()
                    },
                    'selections': selections,
                })
        else:
            branch = 'watershed_dense'
            dense['images'] += 1
            masks_by_instance, _ = watershed_candidates(
                normalized, points, labels, (1.0,))
            for obj, masks in zip(objects, masks_by_instance):
                iou = rotated_iou(masks[0], obj['polygon']) if masks else 0.0
                dense['instances'] += 1
                dense['iou_sum'] += iou
                dense['per_class'][obj['class_name']][0] += 1
                dense['per_class'][obj['class_name']][1] += iou
                object_records.append({
                    'class_name': obj['class_name'], 'watershed_iou': iou})

        image_records.append({
            'image_id': annotation_path.stem, 'branch': branch,
            'instance_count': len(objects), 'objects': object_records})
        if image_index % args.progress_every == 0 or image_index == len(annotations):
            print(json.dumps({
                'done': image_index, 'total': len(annotations),
                'sam_sparse_images': sparse['images'],
                'watershed_dense_images': dense['images'],
                'elapsed_seconds': round(time.perf_counter() - started, 1),
            }), flush=True)

    count = sparse['instances']
    baseline_iou = sparse['baseline_sum'] / count if count else 0.0
    finalized_strategies = {
        name: finalize_stats(values, baseline_iou)
        for name, values in strategy_stats.items()
    }
    for values in finalized_strategies.values():
        for class_name, class_values in values['per_class'].items():
            class_count, class_sum = sparse['per_class_baseline'][class_name]
            class_baseline = class_sum / class_count if class_count else 0.0
            class_values['baseline_iou'] = class_baseline
            class_values['selected_gain'] = (
                class_values['selected_iou'] - class_baseline)
    sparse_summary = {
        'images': sparse['images'], 'instances': count,
        'baseline_iou': baseline_iou,
        'raw9_oracle_iou': sparse['raw_oracle_sum'] / count if count else 0.0,
        'stage1_oracle_iou': (
            sparse['stage1_oracle_sum'] / count if count else 0.0),
        'raw9_oracle_views': dict(sparse['raw_oracle_views']),
        'stage1_oracle_views': dict(sparse['stage1_oracle_views']),
        'strategies': finalized_strategies,
    }
    dense_count = dense['instances']
    dense_summary = {
        'images': dense['images'], 'instances': dense_count,
        'watershed_iou': dense['iou_sum'] / dense_count if dense_count else 0.0,
        'per_class': {
            name: {'instances': values[0],
                   'watershed_iou': values[1] / values[0]}
            for name, values in dense['per_class'].items()
        },
    }
    output = {
        'protocol': {
            'samples': len(annotations), 'seed': args.seed,
            'sample_ids': [path.stem for path in annotations],
            'stage1': 'original filter_masks independently within each view',
            'reliability_weights': {'cross': 0.5, 'sam': 0.3, 'center': 0.2},
            'strategies': STRATEGIES,
        },
        'summary': {'sam_sparse': sparse_summary,
                    'watershed_dense': dense_summary},
        'images': image_records,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    concise = dict(sparse_summary)
    concise.pop('strategies')
    concise['strategies'] = {
        name: {key: values[key] for key in (
            'selected_iou', 'selected_gain', 'replacement_rate',
            'wrong_replacement_rate_all',
            'wrong_replacement_rate_replaced')}
        for name, values in sparse_summary['strategies'].items()
    }
    print(json.dumps({'sam_sparse': concise,
                      'watershed_dense': dense_summary}, indent=2), flush=True)
    print(f'Wrote {output_path}', flush=True)


if __name__ == '__main__':
    main()
