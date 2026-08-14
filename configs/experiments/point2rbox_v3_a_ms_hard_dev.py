_base_ = '../point2rbox_v3/point2rbox_v3-1x-dotav1-0.py'

custom_imports = dict(
    imports=[
        'projects.point2rbox_v3_multiscale.multiscale_voronoi_loss'
    ],
    allow_failed_imports=False)

model = dict(
    bbox_head=dict(
        loss_voronoi=dict(
            type='MultiScaleVoronoiWatershedLoss',
            enabled=True,
            scales=(0.75, 1.0, 1.25),
            max_masks_per_scale=3,
            dedup_iou_thr=0.9)))
