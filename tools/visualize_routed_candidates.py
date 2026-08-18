#!/usr/bin/env python3
"""Visualize C0/C1 with sparse-SAM/dense-watershed routing."""

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.point2rbox_v3_multiscale.candidate_pool import (  # noqa: E402
    MultiScaleSAMCandidatePool,
)
from tools.evaluate_candidate_oracle import (  # noqa: E402
    CONFIGS, choose_samples, load_annotation, locate_image,
    training_image_views, watershed_candidates,
)
from tools.visualize_sam_oracle import add_title, render_oracle  # noqa: E402

COMPARE_CONFIGS = {name: CONFIGS[name] for name in ('C0', 'C1')}


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data/split_ss_dota/trainval')
    parser.add_argument('--sam-checkpoint', default='mobile_sam.pt')
    parser.add_argument('--samples', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sam-instance-thr', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', default='work_dirs/routed_oracle_vis_500_seed0')
    parser.add_argument('--progress-every', type=int, default=10)
    parser.add_argument('--sam-instance-batch-size', type=int, default=16)
    return parser.parse_args()


def protocol(args):
    return {
        'samples': args.samples, 'seed': args.seed,
        'sam_instance_thr': args.sam_instance_thr,
        'iou': 'min-area candidate OBB vs GT OBB rotated IoU',
        'configs': COMPARE_CONFIGS,
    }


def wrap_watershed(masks_by_instance, radii, clusters_by_instance):
    wrapped = []
    for masks, clusters in zip(masks_by_instance, clusters_by_instance):
        items = []
        for index, (mask, cluster_size) in enumerate(zip(masks, clusters)):
            # Radius representatives prefer 1.0, then nearest alternatives.
            ordered = sorted(radii, key=lambda value: abs(np.log(value)))
            source = ordered[min(index, len(ordered) - 1)]
            items.append(SimpleNamespace(
                mask=mask, source_scale=source,
                cluster_scales=[source] * cluster_size))
        wrapped.append(items)
    return wrapped


def main():
    args = args_parser()
    root, out = Path(args.data_root), Path(args.output_dir)
    (out / 'comparison').mkdir(parents=True, exist_ok=True)
    for name in COMPARE_CONFIGS:
        (out / name / 'overlay').mkdir(parents=True, exist_ok=True)
        (out / name / 'oracle_mask').mkdir(parents=True, exist_ok=True)

    annotations = choose_samples(root / 'annfiles', args.samples, args.seed)
    from mobile_sam import SamPredictor, sam_model_registry
    sam = sam_model_registry['vit_t'](checkpoint=args.sam_checkpoint)
    sam.to(args.device).eval()
    predictor = SamPredictor(sam)
    pools = {
        scales: MultiScaleSAMCandidatePool(
            scales=scales, max_masks_per_scale=3, dedup_iou_thr=0.9,
            instance_batch_size=args.sam_instance_batch_size)
        for scales in {value[0] for value in COMPARE_CONFIGS.values()}
    }
    results, started = [], time.perf_counter()
    for image_index, ann_path in enumerate(annotations, 1):
        objects = load_annotation(ann_path)
        image_path = locate_image(root / 'images', ann_path.stem)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        normalized, sam_image = training_image_views(bgr)
        centers = np.asarray([item['center'] for item in objects], np.float32)
        labels = np.asarray([item['class_id'] for item in objects], np.int64)
        branch = ('sam_sparse' if len(objects) <= args.sam_instance_thr
                  else 'watershed_dense')
        generated = {}
        if branch == 'sam_sparse':
            for scales, pool in pools.items():
                begin = time.perf_counter()
                candidates = pool.generate(
                    predictor, sam_image, centers, labels,
                    sample_rules={'filter_pairs': [(3, 10, 200)]})
                generated[scales] = (candidates, time.perf_counter() - begin)
        else:
            for radii in {value[1] for value in COMPARE_CONFIGS.values()}:
                begin = time.perf_counter()
                masks, clusters = watershed_candidates(
                    normalized, centers, labels, radii)
                generated[radii] = (
                    wrap_watershed(masks, radii, clusters),
                    time.perf_counter() - begin)

        image_result = {'image_id': ann_path.stem, 'branch': branch,
                        'instances': len(objects), 'configs': {}}
        panels = []
        for name, (scales, radii) in COMPARE_CONFIGS.items():
            key = scales if branch == 'sam_sparse' else radii
            candidates, elapsed = generated[key]
            canvas, label_mask, records = render_oracle(bgr, objects, candidates)
            overlay = out / name / 'overlay' / f'{ann_path.stem}.jpg'
            mask_path = out / name / 'oracle_mask' / f'{ann_path.stem}.png'
            cv2.imwrite(str(overlay), canvas)
            cv2.imwrite(str(mask_path), label_mask)
            values = [item['oracle_iou'] for item in records]
            mean_iou = float(np.mean(values))
            recall = float(np.mean(np.asarray(values) >= 0.5))
            image_result['configs'][name] = {
                'seconds': elapsed, 'mean_oracle_iou': mean_iou,
                'recall_at_05': recall, 'objects': records,
                'overlay_path': str(overlay), 'oracle_mask_path': str(mask_path),
            }
            panels.append(add_title(
                canvas, f'{name} {branch} | IoU={mean_iou:.3f} R@0.5={recall:.3f}'))
        comparison = np.hstack(panels)
        comparison_path = out / 'comparison' / f'{ann_path.stem}.jpg'
        cv2.imwrite(str(comparison_path), comparison)
        image_result['comparison_path'] = str(comparison_path)
        results.append(image_result)
        (out / 'results.json').write_text(json.dumps({
            'protocol': protocol(args), 'images': results,
        }, ensure_ascii=False, indent=2))
        if image_index % args.progress_every == 0 or image_index == len(annotations):
            print(json.dumps({'done': image_index, 'total': len(annotations),
                              'elapsed_seconds': round(time.perf_counter()-started, 1)}),
                  flush=True)
    print(f'Wrote {out}', flush=True)


if __name__ == '__main__':
    main()
