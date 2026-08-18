#!/usr/bin/env python3
"""Run single/multi-scale MobileSAM on every selected image and visualize IoU."""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load this standalone module without importing the project's package __init__.
# The latter pulls in the full MMDetection stack, which is unnecessary for this
# offline SAM-only evaluation and makes the script depend on its MMCV version.
_POOL_PATH = REPO_ROOT / 'projects/point2rbox_v3_multiscale/candidate_pool.py'
_POOL_SPEC = importlib.util.spec_from_file_location(
    'point2rbox_multiscale_candidate_pool', _POOL_PATH)
_POOL_MODULE = importlib.util.module_from_spec(_POOL_SPEC)
sys.modules[_POOL_SPEC.name] = _POOL_MODULE
_POOL_SPEC.loader.exec_module(_POOL_MODULE)
MultiScaleSAMCandidatePool = _POOL_MODULE.MultiScaleSAMCandidatePool
from tools.evaluate_candidate_oracle import (  # noqa: E402
    choose_samples, load_annotation, locate_image, rotated_iou,
    training_image_views,
)


SETTINGS = {
    'single_1.0': (1.0,),
    'multi_0.75_1.0_1.25': (0.75, 1.0, 1.25),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data/split_ss_dota/trainval')
    parser.add_argument('--sam-checkpoint', default='mobile_sam.pt')
    parser.add_argument('--samples', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-instances', type=int, default=None,
                        help='Keep only non-empty images with at most this many GT instances')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', default='work_dirs/sam_oracle_vis_200_seed0')
    parser.add_argument('--progress-every', type=int, default=10)
    parser.add_argument('--settings', nargs='+', choices=tuple(SETTINGS),
                        default=list(SETTINGS))
    parser.add_argument('--sam-instance-batch-size', type=int, default=16)
    parser.add_argument('--view-strategy', choices=('whole_resize', 'asymmetric'),
                        default='whole_resize')
    parser.add_argument('--zoom-grid-size', type=int, default=2)
    parser.add_argument('--crop-image-batch-size', type=int, default=4)
    parser.add_argument('--skip-comparison', action='store_true')
    parser.add_argument('--resume', action='store_true',
                        help='Keep existing results and process only new IDs')
    return parser.parse_args()


def color_for(index):
    rng = np.random.default_rng(index + 2026)
    return tuple(int(value) for value in rng.integers(70, 240, size=3))


def candidate_box(mask):
    points = np.column_stack(np.nonzero(mask))[..., ::-1].astype(np.float32)
    if len(points) < 3:
        return None
    return cv2.boxPoints(cv2.minAreaRect(points)).round().astype(np.int32)


def render_oracle(bgr, objects, candidates):
    overlay = bgr.copy()
    label_mask = np.zeros(bgr.shape[:2], dtype=np.uint16)
    records = []
    chosen = []
    for index, (obj, items) in enumerate(zip(objects, candidates)):
        masks = [item.mask for item in items]
        ious = [rotated_iou(mask, obj['polygon']) for mask in masks]
        best_index = int(np.argmax(ious)) if ious else None
        best_iou = float(ious[best_index]) if ious else 0.0
        best_mask = masks[best_index] if best_index is not None else None
        chosen.append(best_mask)
        if best_mask is not None:
            color = color_for(index)
            overlay[best_mask] = (
                0.55 * overlay[best_mask] + 0.45 * np.asarray(color)
            ).astype(np.uint8)
            label_mask[best_mask] = index + 1
        records.append({
            'instance_index': index,
            'class_name': obj['class_name'],
            'difficulty': obj['difficulty'],
            'candidate_count': len(items),
            'oracle_iou': best_iou,
            'best_candidate_index': best_index,
            'best_source_scale': (
                items[best_index].source_scale if best_index is not None else None),
            'best_cluster_scales': (
                items[best_index].cluster_scales if best_index is not None else []),
        })

    canvas = cv2.addWeighted(bgr, 0.25, overlay, 0.75, 0)
    for index, (obj, mask, record) in enumerate(zip(objects, chosen, records)):
        gt = obj['polygon'].round().astype(np.int32)
        cv2.polylines(canvas, [gt], True, (0, 255, 0), 2, cv2.LINE_AA)
        if mask is not None:
            box = candidate_box(mask)
            if box is not None:
                cv2.polylines(canvas, [box], True, (0, 220, 255), 2,
                              cv2.LINE_AA)
        x, y = gt[0]
        text = f"{index}:{obj['class_name']} IoU={record['oracle_iou']:.3f}"
        cv2.putText(canvas, text, (int(x), max(14, int(y) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(canvas, text, (int(x), max(14, int(y) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return canvas, label_mask, records


def add_title(image, title):
    header = np.full((42, image.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(header, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([header, image])


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    (output_dir / 'comparison').mkdir(parents=True, exist_ok=True)
    selected_settings = {name: SETTINGS[name] for name in args.settings}
    for name in selected_settings:
        (output_dir / name / 'overlay').mkdir(parents=True, exist_ok=True)
        (output_dir / name / 'oracle_mask').mkdir(parents=True, exist_ok=True)

    annotations = choose_samples(data_root / 'annfiles', args.samples, args.seed)
    if args.max_instances is not None:
        annotations = [
            path for path in annotations
            if len(load_annotation(path)) <= args.max_instances
        ]
    import torch
    from mobile_sam import SamPredictor, sam_model_registry
    sam = sam_model_registry['vit_t'](checkpoint=args.sam_checkpoint)
    sam.to(args.device).eval()
    predictor = SamPredictor(sam)
    pools = {
        name: MultiScaleSAMCandidatePool(
            scales=scales, max_masks_per_scale=3, dedup_iou_thr=0.9,
            instance_batch_size=args.sam_instance_batch_size,
            view_strategy=args.view_strategy,
            zoom_grid_size=args.zoom_grid_size,
            crop_image_batch_size=args.crop_image_batch_size)
        for name, scales in selected_settings.items()
    }

    results_path = output_dir / 'results.json'
    results = []
    if args.resume and results_path.exists():
        results = json.loads(results_path.read_text()).get('images', [])
    completed_ids = {item['image_id'] for item in results}
    pending_annotations = [
        path for path in annotations if path.stem not in completed_ids
    ]
    print(json.dumps({
        'target_images': len(annotations),
        'already_completed': len(completed_ids),
        'pending': len(pending_annotations),
    }), flush=True)
    started = time.perf_counter()
    for pending_index, annotation_path in enumerate(pending_annotations, 1):
        objects = load_annotation(annotation_path)
        image_path = locate_image(data_root / 'images', annotation_path.stem)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        _, sam_image = training_image_views(bgr)
        centers = np.asarray([obj['center'] for obj in objects], dtype=np.float32)
        labels = np.asarray([obj['class_id'] for obj in objects], dtype=np.int64)
        image_record = {
            'image_id': annotation_path.stem,
            'image_path': str(image_path),
            'instances': len(objects),
            'settings': {},
        }
        panels = []
        for name, pool in pools.items():
            begin = time.perf_counter()
            candidates = pool.generate(
                predictor, sam_image, centers, labels,
                sample_rules={'filter_pairs': [(3, 10, 200)]})
            elapsed = time.perf_counter() - begin
            canvas, label_mask, records = render_oracle(
                bgr, objects, candidates)
            overlay_path = output_dir / name / 'overlay' / f'{annotation_path.stem}.jpg'
            mask_path = output_dir / name / 'oracle_mask' / f'{annotation_path.stem}.png'
            cv2.imwrite(str(overlay_path), canvas)
            cv2.imwrite(str(mask_path), label_mask)
            mean_iou = float(np.mean([item['oracle_iou'] for item in records]))
            recall = float(np.mean([item['oracle_iou'] >= 0.5 for item in records]))
            image_record['settings'][name] = {
                'seconds': elapsed,
                'mean_oracle_iou': mean_iou,
                'recall_at_05': recall,
                'overlay_path': str(overlay_path),
                'oracle_mask_path': str(mask_path),
                'objects': records,
            }
            panels.append(add_title(
                canvas, f'{name} | mean Oracle IoU={mean_iou:.3f} | R@0.5={recall:.3f}'))
        if not args.skip_comparison:
            comparison = np.hstack(panels)
            comparison_path = output_dir / 'comparison' / f'{annotation_path.stem}.jpg'
            cv2.imwrite(str(comparison_path), comparison)
            image_record['comparison_path'] = str(comparison_path)
        results.append(image_record)
        should_checkpoint = (
            pending_index % args.progress_every == 0 or
            pending_index == len(pending_annotations))
        if should_checkpoint:
            # Rewriting an ever-growing JSON after every image is quadratic
            # on full datasets; checkpoint at the reporting interval instead.
            results_path.write_text(json.dumps({
                'protocol': {
                    'samples_requested': args.samples, 'seed': args.seed,
                    'max_instances': args.max_instances,
                    'settings': selected_settings,
                    'sam_instance_batch_size': args.sam_instance_batch_size,
                    'view_strategy': args.view_strategy,
                    'zoom_grid_size': args.zoom_grid_size,
                    'crop_image_batch_size': args.crop_image_batch_size},
                'images': results,
            }, indent=2, ensure_ascii=False))
            print(json.dumps({
                'added': pending_index, 'pending_total': len(pending_annotations),
                'completed_total': len(results), 'target_total': len(annotations),
                'elapsed_seconds': round(time.perf_counter() - started, 1),
            }), flush=True)

    print(f'Wrote visualizations and IoUs to {output_dir}', flush=True)


if __name__ == '__main__':
    main()
