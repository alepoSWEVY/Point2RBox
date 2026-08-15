_base_ = 'point2rbox_v3_a_ms_sam_mr_watershed_hard_dev.py'

custom_imports = dict(
    imports=['projects.point2rbox_v3_soft_pgdm'],
    allow_failed_imports=False)

model = dict(
    bbox_head=dict(
        loss_voronoi=dict(
            type='SoftPGDMLoss',
            sam_candidate_pool_enabled=True,
            watershed_candidate_pool_enabled=True,
            quality_beta=0.5,
            quality_delta=0.5,
            temperature=1.0,
            instance_gate_enabled=False)))
