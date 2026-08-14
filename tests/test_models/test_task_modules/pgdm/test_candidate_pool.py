import numpy as np

from projects.point2rbox_v3_multiscale.candidate_pool import (
    Candidate, MultiScaleSAMCandidatePool, ScaleTransform,
    deduplicate_candidates)


class FakePredictor:

    def __init__(self):
        self.image_shapes = []
        self.image_shape = None

    def set_image(self, image):
        self.image_shape = image.shape[:2]
        self.image_shapes.append(self.image_shape)

    def predict(self, point_coords, point_labels, box, multimask_output):
        height, width = self.image_shape
        masks = np.zeros((3, height, width), dtype=bool)
        center_x, center_y = np.rint(point_coords[0]).astype(int)
        center_x = np.clip(center_x, 4, width - 5)
        center_y = np.clip(center_y, 4, height - 5)
        masks[0, center_y - 4:center_y + 5,
              center_x - 4:center_x + 5] = True
        masks[1] = masks[0]
        masks[2, center_y - 2:center_y + 3,
              center_x - 2:center_x + 3] = True
        return masks, np.array([0.8, 0.9, 0.7]), None


def test_scale_transform_restores_original_shape():
    transform = ScaleTransform.from_scale((101, 203, 3), 0.75)
    points = np.array([[0.0, 0.0], [202.0, 100.0]])
    mapped = transform.map_points(points)
    assert transform.scaled_size == (76, 152)
    assert np.allclose(mapped[-1],
                       [202.0 * 152 / 203, 100.0 * 76 / 101])
    restored = transform.restore_mask(
        np.ones(transform.scaled_size, dtype=bool))
    assert restored.shape == (101, 203)


def test_dedup_keeps_highest_score_and_merges_scale_sources():
    mask = np.ones((8, 8), dtype=bool)
    candidates = [
        Candidate(mask, 0.6, 0.75, 0, (0.75, )),
        Candidate(mask.copy(), 0.9, 1.0, 1, (1.0, )),
    ]
    result = deduplicate_candidates(candidates, 0.9)
    assert len(result) == 1
    assert result[0].score == 0.9
    assert result[0].cluster_scales == (0.75, 1.0)


def test_candidate_pool_maps_and_deduplicates_all_scales():
    predictor = FakePredictor()
    pool = MultiScaleSAMCandidatePool()
    image = np.zeros((64, 80, 3), dtype=np.uint8)
    points = np.array([[40.0, 32.0]], dtype=np.float32)
    labels = np.array([3])
    result = pool.generate(predictor, image, points, labels)
    assert predictor.image_shapes == [(48, 60), (64, 80), (80, 100)]
    assert result[0]
    assert all(candidate.mask.shape == (64, 80)
               for candidate in result[0])
    assert all(candidate.mask.any() for candidate in result[0])
