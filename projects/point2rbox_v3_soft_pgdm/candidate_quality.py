import math

import numpy as np
import torch


def mask_iou(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union else 0.0


def _candidate_sources(candidate):
    if hasattr(candidate, 'cluster_scales'):
        return tuple(candidate.cluster_scales), candidate.source_scale
    if hasattr(candidate, 'cluster_radii'):
        return tuple(candidate.cluster_radii), candidate.source_radius
    raise TypeError('Candidate has no scale or radius source metadata')


def cross_source_consistency(candidate, candidates):
    """Average best IoU against every other available source view."""
    candidate_sources, primary_source = _candidate_sources(candidate)
    all_sources = sorted({
        source
        for item in candidates
        for source in _candidate_sources(item)[0]
    })
    other_sources = [source for source in all_sources
                     if source != primary_source]
    if not other_sources:
        return 0.0

    best_ious = []
    for source in other_sources:
        if source in candidate_sources:
            best_ious.append(1.0)
            continue
        matching = [
            item for item in candidates
            if source in _candidate_sources(item)[0]
        ]
        best_ious.append(max(
            (mask_iou(candidate.mask, item.mask) for item in matching),
            default=0.0))
    return float(np.mean(best_ious))


def instance_separation(mask, points, labels, instance_index):
    """Penalize masks containing additional points of the same class."""
    height, width = mask.shape
    current_label = int(labels[instance_index])
    other_count = 0
    for point_index, (point, label) in enumerate(zip(points, labels)):
        if point_index == instance_index or int(label) != current_label:
            continue
        x, y = np.rint(point).astype(np.int64)
        if 0 <= x < width and 0 <= y < height and mask[y, x]:
            other_count += 1
    return 1.0 / (1.0 + other_count)


def candidate_quality_scores(candidates,
                             points,
                             labels,
                             instance_index,
                             beta=0.5,
                             delta=0.5):
    denominator = beta + delta
    if denominator <= 0:
        raise ValueError('beta + delta must be positive')
    qualities = []
    for candidate in candidates:
        scale_quality = cross_source_consistency(candidate, candidates)
        separation_quality = instance_separation(
            candidate.mask, points, labels, instance_index)
        qualities.append((beta * scale_quality +
                          delta * separation_quality) / denominator)
    return qualities


def soft_candidate_weights(qualities, temperature, device):
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    quality_tensor = torch.as_tensor(
        qualities, dtype=torch.float32, device=device)
    return torch.softmax(quality_tensor / temperature, dim=0).detach()


def instance_quality_gate(qualities,
                          probabilities,
                          quality_threshold=0.5,
                          smoothness=0.1,
                          enabled=True):
    if not qualities:
        return probabilities.new_tensor(0.0)
    if not enabled:
        return probabilities.new_tensor(1.0)
    if smoothness <= 0:
        raise ValueError('gate smoothness must be positive')

    absolute_quality = probabilities.new_tensor(max(qualities))
    quality_gate = torch.sigmoid(
        (absolute_quality - quality_threshold) / smoothness)
    if len(probabilities) == 1:
        normalized_entropy = probabilities.new_tensor(0.0)
    else:
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
        normalized_entropy = entropy / math.log(len(probabilities))
    return (quality_gate * (1.0 - normalized_entropy)).detach()
