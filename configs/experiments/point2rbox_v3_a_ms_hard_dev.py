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
            instance_batch_size=16,
            view_strategy='asymmetric',
            zoom_grid_size=2,
            crop_image_batch_size=4,
            selector=dict(
                center_weight=1.0,
                sam_weight=0.0,
                cross_view_weight=0.0,
                replacement_margin=0.05,
                min_mask_area=16,
                max_image_coverage=0.95,
                crop_boundary_width=3,
                crop_boundary_ratio=0.02,
                crop_boundary_penalty=0.5),
            dedup_iou_thr=0.9)))
