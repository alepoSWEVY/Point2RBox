_base_ = 'point2rbox_v3_a_ms_sam_mr_watershed_hard_dev.py'

custom_imports = dict(
    imports=[
        'projects.point2rbox_v3_multiradius_watershed.multiradius_pgdm_loss',
        'projects.point2rbox_v3_soft_pgdm',
    ],
    allow_failed_imports=False)

model = dict(
    type='Point2RBoxV3EMATeacher',
    ema_teacher=dict(
        enabled=True,
        start_epoch=6,
        momentum=0.999,
        use_for_quality=False,
        use_direct_box_loss=False))

custom_hooks = [
    dict(type='mmdet.SetEpochInfoHook'),
    dict(
        type='ProgressiveEMAHook',
        start_epoch=6,
        momentum=0.999,
        priority='LOW')
]
