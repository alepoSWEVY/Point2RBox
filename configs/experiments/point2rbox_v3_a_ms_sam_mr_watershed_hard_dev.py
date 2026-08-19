_base_ = '../point2rbox_v3/point2rbox_v3-1x-dotav1-0.py'

custom_imports = dict(
    imports=[
        'projects.point2rbox_v3_multiradius_watershed.multiradius_pgdm_loss'
    ],
    allow_failed_imports=False)

model = dict(
    bbox_head=dict(
        loss_voronoi=dict(
            type='MultiRadiusPGDMLoss',
            sam_candidate_pool_enabled=True,
            watershed_candidate_pool_enabled=True,
            sam_scales=(0.75, 1.0, 1.25),
            sam_max_masks_per_scale=3,
            sam_instance_batch_size=16,
            sam_dedup_iou_thr=0.9,
            watershed_radius_multipliers=(0.75, 1.0, 1.25),
            watershed_dedup_iou_thr=0.9)))
