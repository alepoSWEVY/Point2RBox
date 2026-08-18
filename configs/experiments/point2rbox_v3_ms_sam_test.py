_base_ = './point2rbox_v3_a_ms_sam_mr_watershed_hard_dev.py'

# Multi-scale SAM is used only as inference-time geometry refinement here.
# Watershed, multi-radius watershed, EMA, and soft-PGDM are not enabled.
model = dict(
    bbox_head=dict(
        loss_voronoi=dict(
            # C1: sparse multi-scale SAM, dense original watershed.
            sam_candidate_pool_enabled=True,
            watershed_candidate_pool_enabled=False,
            sam_scales=(0.75, 1.0, 1.25),
            watershed_radius_multipliers=(1.0,))),
    sam_test_cfg=dict(
        enabled=True,
        sam_instance_thr=4,
        score_thr=0.3,
        max_per_img=0,
        min_mask_area=4,
        report_interval=50))

# SAM refinement is expensive; keep inference deterministic and use one image
# per worker process/GPU.
test_dataloader = dict(batch_size=1)
test_evaluator = dict(outfile_prefix=
                      './work_dirs/test_epoch12_ms_sam/Task1')
