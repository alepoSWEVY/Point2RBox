import numpy as np
import torch

from mmrotate.models.losses.point2rbox_v2_loss import gwd_sigma_loss
from mmrotate.registry import MODELS
from projects.point2rbox_v3_multiradius_watershed.multiradius_pgdm_loss import (
    MultiRadiusPGDMLoss)

from .candidate_quality import (candidate_quality_scores,
                                instance_quality_gate,
                                soft_candidate_weights)


@MODELS.register_module()
class SoftPGDMLoss(MultiRadiusPGDMLoss):
    """Soft candidate-weighted PGDM without Teacher participation."""

    def __init__(self,
                 quality_beta=0.5,
                 quality_delta=0.5,
                 temperature=1.0,
                 instance_gate_enabled=False,
                 quality_threshold=0.5,
                 gate_smoothness=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        if quality_beta < 0 or quality_delta < 0:
            raise ValueError('quality weights must be non-negative')
        if quality_beta + quality_delta <= 0:
            raise ValueError('quality weights must have a positive sum')
        if temperature <= 0:
            raise ValueError('temperature must be positive')
        self.quality_beta = quality_beta
        self.quality_delta = quality_delta
        self.temperature = temperature
        self.instance_gate_enabled = instance_gate_enabled
        self.quality_threshold = quality_threshold
        self.gate_smoothness = gate_smoothness

    def _soft_candidate_loss(self, mu, sigma, label, candidates_by_instance):
        eigenvalues, eigenvectors = torch.linalg.eigh(sigma)
        points = mu.detach().cpu().numpy()
        labels = label.detach().cpu().numpy()
        numerator = sigma.sum() * 0
        denominator = sigma.new_tensor(0.0)
        first_candidate = next(
            (candidates[0] for candidates in candidates_by_instance
             if candidates), None)
        height, width = first_candidate.mask.shape \
            if first_candidate is not None else (0, 0)
        markers = torch.full(
            (height, width), len(mu) + 1, dtype=torch.int32,
            device=mu.device) if height and width else None

        for instance_index, candidates in enumerate(candidates_by_instance):
            if not candidates:
                continue
            qualities = candidate_quality_scores(
                candidates,
                points,
                labels,
                instance_index,
                beta=self.quality_beta,
                delta=self.quality_delta)
            probabilities = soft_candidate_weights(
                qualities, self.temperature, mu.device)
            gate = instance_quality_gate(
                qualities,
                probabilities,
                quality_threshold=self.quality_threshold,
                smoothness=self.gate_smoothness,
                enabled=self.instance_gate_enabled)

            candidate_losses = []
            for candidate in candidates:
                mask = torch.from_numpy(candidate.mask).to(
                    device=mu.device, dtype=torch.bool)
                coordinates = mask.nonzero()[:, (1, 0)].float()
                if len(coordinates) == 0:
                    candidate_losses.append(eigenvalues[instance_index].sum()
                                            * 0)
                    continue
                centered = coordinates - mu[instance_index].detach()
                rotated = eigenvectors[instance_index].detach().T.matmul(
                    centered[:, :, None])[:, :, 0]
                target = torch.stack((
                    torch.max(torch.abs(rotated[:, 0])),
                    torch.max(torch.abs(rotated[:, 1])))).square().detach()
                candidate_losses.append(gwd_sigma_loss(
                    torch.diag_embed(eigenvalues[instance_index]).unsqueeze(0),
                    torch.diag_embed(target).unsqueeze(0),
                    reduction='none').reshape(-1).mean())

            loss = torch.sum(probabilities * torch.stack(candidate_losses))
            numerator = numerator + gate * loss
            denominator = denominator + gate
            if markers is not None:
                best_index = int(np.argmax(qualities))
                best_mask = torch.from_numpy(
                    candidates[best_index].mask).to(mu.device)
                markers[best_mask] = instance_index + 1

        self.vis = (markers.clone(), markers) if markers is not None else None
        return self.loss_weight * numerator / denominator.clamp_min(1e-6)

    def _forward_multiscale_sam(self, mu, sigma, label, image):
        image_np = self._image_to_uint8(image)
        predictor = self._build_predictor(mu.device)
        candidates = self.candidate_pool.generate(
            predictor,
            image_np,
            mu.detach().cpu().numpy(),
            label.detach().cpu().numpy(),
            sample_rules=self.sam_sample_rules)
        return self._soft_candidate_loss(mu, sigma, label, candidates)

    def _forward_multiradius_watershed(self, mu, sigma, label, image,
                                       pos_thres, neg_thres, voronoi):
        candidates = self.watershed_candidate_pool.generate(
            image,
            mu,
            sigma,
            label,
            pos_thres,
            neg_thres,
            down_sample=self.down_sample,
            default_sigma=self.default_sigma,
            voronoi=voronoi)
        return self._soft_candidate_loss(mu, sigma, label, candidates)
