_base_ = 'point2rbox_v3_soft_pgdm_s1_dev.py'

model = dict(
    bbox_head=dict(
        loss_voronoi=dict(
            instance_gate_enabled=True,
            quality_threshold=0.5,
            gate_smoothness=0.1)))
