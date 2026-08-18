#!/usr/bin/env python3
"""Offline oracle evaluation for SAM/watershed candidate ablations on DOTA.

This mirrors the current Point2RBox-v3 routing: images with at most four
instances use point-prompted MobileSAM; denser images use standard Gaussian
Voronoi watershed.  No detector is built and no weights are updated.
"""

import argparse
import importlib.util
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_POOL_PATH = REPO_ROOT / 'projects/point2rbox_v3_multiscale/candidate_pool.py'
_POOL_SPEC = importlib.util.spec_from_file_location(
    'point2rbox_multiscale_candidate_pool', _POOL_PATH)
if _POOL_SPEC.name in sys.modules:
    _POOL_MODULE = sys.modules[_POOL_SPEC.name]
else:
    _POOL_MODULE = importlib.util.module_from_spec(_POOL_SPEC)
    sys.modules[_POOL_SPEC.name] = _POOL_MODULE
    _POOL_SPEC.loader.exec_module(_POOL_MODULE)
MultiScaleSAMCandidatePool = _POOL_MODULE.MultiScaleSAMCandidatePool


CLASSES = (
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
    'harbor', 'swimming-pool', 'helicopter')
CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}
CONFIGS = {
    'C0': ((1.0,), (1.0,)),
    'C1': ((0.75, 1.0, 1.25), (1.0,)),
    'C2': ((1.0,), (0.75, 1.0, 1.25)),
    'C3': ((0.75, 1.0, 1.25), (0.75, 1.0, 1.25)),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data/split_ss_dota/trainval')
    parser.add_argument('--sam-checkpoint', default='mobile_sam.pt')
    parser.add_argument('--samples', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sam-instance-thr', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', default='work_dirs/candidate_eval_500_seed0.json')
    parser.add_argument('--configs', nargs='+', choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument('--progress-every', type=int, default=10)
    return parser.parse_args()


def load_annotation(path):
    objects = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 10 or fields[8] not in CLASS_TO_ID:
            continue
        polygon = np.asarray(fields[:8], dtype=np.float32).reshape(4, 2)
        objects.append({
            'polygon': polygon,
            'center': polygon.mean(axis=0),
            'class_id': CLASS_TO_ID[fields[8]],
            'class_name': fields[8],
            'difficulty': int(fields[9]),
        })
    return objects


def choose_samples(annotation_dir, count, seed):
    paths = sorted(annotation_dir.glob('*.txt'))
    rng = random.Random(seed)
    rng.shuffle(paths)
    # Match the training dataset's filter_empty_gt=True behavior.
    selected = []
    for path in paths:
        if load_annotation(path):
            selected.append(path)
            if len(selected) == count:
                break
    return sorted(selected, key=lambda path: path.name)


def locate_image(image_dir, stem):
    for suffix in ('.png', '.jpg', '.jpeg', '.tif', '.bmp'):
        path = image_dir / f'{stem}{suffix}'
        if path.exists():
            return path
    raise FileNotFoundError(f'No image for annotation {stem}')


def training_image_views(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    mean = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)
    normalized = (rgb - mean) / std
    low, high = float(normalized.min()), float(normalized.max())
    display = np.zeros_like(normalized, dtype=np.uint8) if high == low else (
        (normalized - low) / (high - low) * 255).astype(np.uint8)
    return normalized, display


def dedup_masks(masks, threshold=0.9):
    kept = []
    clusters = []
    for mask in masks:
        match = None
        for index, representative in enumerate(kept):
            union = np.logical_or(mask, representative).sum()
            iou = np.logical_and(mask, representative).sum() / union if union else 0
            if iou > threshold:
                match = index
                break
        if match is None:
            kept.append(mask)
            clusters.append(1)
        else:
            clusters[match] += 1
    return kept, clusters


def watershed_candidates(normalized, centers, labels, radii):
    height, width = normalized.shape[:2]
    down_sample = 2
    small_h, small_w = height // down_sample, width // down_sample
    # Match torch.linspace(0, size, size) and meshgrid(indexing='xy').
    xs = np.linspace(0, small_w, small_w, dtype=np.float32)
    ys = np.linspace(0, small_h, small_h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    centers_small = np.round(centers / down_sample)
    pos = np.full(15, 0.994, dtype=np.float32)
    neg = np.full(15, 0.005, dtype=np.float32)
    pos[[2, 11]], neg[[2, 11]] = 0.999, 0.6
    pos[[7, 8, 10, 14]], neg[[7, 8, 10, 14]] = 0.95, 0.005
    by_instance = [[] for _ in centers]
    # Exact global min/max conversion used by the training loss.
    low, high = float(normalized.min()), float(normalized.max())
    image_uint8 = np.zeros_like(normalized, dtype=np.uint8) if high == low else (
        (normalized - low) / (high - low) * 255).astype(np.uint8)
    image_uint8 = cv2.medianBlur(image_uint8, 3)
    for radius in radii:
        variance = 4096.0 * radius * radius / (down_sample * down_sample)
        distributions = []
        for center_x, center_y in centers_small:
            squared = (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2
            distributions.append(np.exp(-0.5 * squared / variance))
        distributions = np.asarray(distributions, dtype=np.float32)
        regions_small = distributions.argmax(axis=0).astype(np.int32)
        values_small = distributions.max(axis=0)
        regions = np.repeat(np.repeat(regions_small, down_sample, 0), down_sample, 1)
        values = cv2.resize(values_small, (width, height), interpolation=cv2.INTER_LINEAR)
        class_map = labels[regions]
        kernel = np.ones((3, 3), dtype=np.float32)
        kernel[1, 1] = -8
        ridges = cv2.filter2D(regions.astype(np.float32), -1, kernel,
                              borderType=cv2.BORDER_CONSTANT) != 0
        markers = regions + 1
        markers[values < pos[class_map]] = 0
        markers[values < neg[class_map]] = len(centers) + 1
        markers[ridges] = len(centers) + 1
        markers = cv2.watershed(image_uint8.copy(), markers.astype(np.int32))
        for index in range(len(centers)):
            mask = markers == index + 1
            if mask.any():
                by_instance[index].append(mask)
    result, cluster_sizes = [], []
    for masks in by_instance:
        kept, clusters = dedup_masks(masks)
        result.append(kept)
        cluster_sizes.append(clusters)
    return result, cluster_sizes


def rotated_iou(mask, gt_polygon):
    points = np.column_stack(np.nonzero(mask))[..., ::-1].astype(np.float32)
    if len(points) < 3:
        return 0.0
    candidate_rect = cv2.minAreaRect(points)
    gt_rect = cv2.minAreaRect(gt_polygon.astype(np.float32))
    area_a = candidate_rect[1][0] * candidate_rect[1][1]
    area_b = gt_rect[1][0] * gt_rect[1][1]
    _, intersection = cv2.rotatedRectangleIntersection(candidate_rect, gt_rect)
    intersection_area = 0.0 if intersection is None else abs(cv2.contourArea(intersection))
    union = area_a + area_b - intersection_area
    return float(intersection_area / union) if union > 0 else 0.0


def empty_stats():
    return {
        'images': 0, 'instances': 0, 'oracle_iou_sum': 0.0, 'recall_05': 0,
        'missing': 0, 'raw_candidates': 0, 'candidates': 0,
        'duplicates_removed': 0, 'seconds': 0.0,
        'per_class': defaultdict(lambda: {
            'instances': 0, 'oracle_iou_sum': 0.0, 'recall_05': 0, 'missing': 0,
        }),
    }


def update_stats(stats, objects, candidates, raw_counts, elapsed):
    stats['images'] += 1
    stats['seconds'] += elapsed
    for obj, masks, raw_count in zip(objects, candidates, raw_counts):
        ious = [rotated_iou(mask, obj['polygon']) for mask in masks]
        oracle = max(ious, default=0.0)
        cls = stats['per_class'][obj['class_name']]
        for target in (stats, cls):
            target['instances'] += 1
            target['oracle_iou_sum'] += oracle
            target['recall_05'] += int(oracle >= 0.5)
            target['missing'] += int(not masks)
        stats['raw_candidates'] += raw_count
        stats['candidates'] += len(masks)
        stats['duplicates_removed'] += raw_count - len(masks)


def finalize(stats):
    result = dict(stats)
    result['per_class'] = dict(stats['per_class'])
    for values in [result, *result['per_class'].values()]:
        count = values['instances']
        values['oracle_iou'] = values.pop('oracle_iou_sum') / count if count else 0.0
        values['recall_at_05'] = values.pop('recall_05') / count if count else 0.0
        values['missing_rate'] = values['missing'] / count if count else 0.0
    raw = result['raw_candidates']
    result['duplicate_rate'] = result['duplicates_removed'] / raw if raw else 0.0
    images = result['images']
    result['seconds_per_image'] = result['seconds'] / images if images else 0.0
    result['candidates_per_image'] = result['candidates'] / images if images else 0.0
    return result


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    annotations = choose_samples(data_root / 'annfiles', args.samples, args.seed)
    if not annotations:
        raise RuntimeError(f'No annotations found below {data_root}')

    import torch
    from mobile_sam import SamPredictor, sam_model_registry
    sam = sam_model_registry['vit_t'](checkpoint=args.sam_checkpoint)
    sam.to(args.device).eval()
    predictor = SamPredictor(sam)
    unique_scales = {CONFIGS[name][0] for name in args.configs}
    pools = {
        scales: MultiScaleSAMCandidatePool(
            scales=scales, max_masks_per_scale=3, dedup_iou_thr=0.9)
        for scales in unique_scales
    }
    stats = {
        name: {subset: empty_stats()
               for subset in ('overall', 'sam_sparse', 'watershed_dense')}
        for name in args.configs
    }
    started = time.perf_counter()
    for image_index, annotation_path in enumerate(annotations, 1):
        objects = load_annotation(annotation_path)
        image_path = locate_image(data_root / 'images', annotation_path.stem)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f'Failed to read {image_path}')
        normalized, sam_image = training_image_views(bgr)
        centers = np.asarray([obj['center'] for obj in objects], dtype=np.float32)
        labels = np.asarray([obj['class_id'] for obj in objects], dtype=np.int64)
        generated = {}
        if len(objects) <= args.sam_instance_thr:
            subset = 'sam_sparse'
            for scales in unique_scales:
                begin = time.perf_counter()
                candidates = pools[scales].generate(
                    predictor, sam_image, centers, labels,
                    sample_rules={'filter_pairs': [(3, 10, 200)]})
                raw_counts = [sum(len(item.cluster_scales) for item in items)
                              for items in candidates]
                masks = [[item.mask for item in items] for items in candidates]
                generated[scales] = (
                    masks, raw_counts, time.perf_counter() - begin)
        else:
            subset = 'watershed_dense'
            unique_radii = {CONFIGS[name][1] for name in args.configs}
            for radii in unique_radii:
                begin = time.perf_counter()
                masks, clusters = watershed_candidates(
                    normalized, centers, labels, radii)
                raw_counts = [sum(items) for items in clusters]
                generated[radii] = (
                    masks, raw_counts, time.perf_counter() - begin)
        for name in args.configs:
            key = CONFIGS[name][0] if subset == 'sam_sparse' else CONFIGS[name][1]
            masks, raw_counts, elapsed = generated[key]
            update_stats(stats[name]['overall'], objects, masks, raw_counts,
                         elapsed)
            update_stats(stats[name][subset], objects, masks, raw_counts,
                         elapsed)
        if image_index % args.progress_every == 0 or image_index == len(annotations):
            print(json.dumps({
                'done': image_index, 'total': len(annotations),
                'elapsed_seconds': round(time.perf_counter() - started, 1),
            }), flush=True)

    output = {
        'protocol': {
            'samples': len(annotations), 'seed': args.seed,
            'sam_instance_thr': args.sam_instance_thr,
            'configs': {name: {'sam_scales': CONFIGS[name][0],
                               'watershed_radii': CONFIGS[name][1]}
                        for name in args.configs},
            'sample_ids': [path.stem for path in annotations],
        },
        'results': {
            name: {subset: finalize(values)
                   for subset, values in subsets.items()}
            for name, subsets in stats.items()
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(json.dumps(result_summary(output), indent=2), flush=True)
    print(f'Wrote {output_path}', flush=True)


def result_summary(output):
    keys = ('oracle_iou', 'recall_at_05', 'duplicate_rate',
            'candidates_per_image', 'seconds_per_image')
    return {
        name: {
            subset: {key: values[key] for key in keys}
            for subset, values in subsets.items()
        }
        for name, subsets in output['results'].items()
    }


if __name__ == '__main__':
    main()
