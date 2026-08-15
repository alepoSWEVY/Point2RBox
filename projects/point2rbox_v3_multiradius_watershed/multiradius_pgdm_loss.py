import numpy as np
import torch

from mmrotate.models.losses.point2rbox_v2_loss import (
    VoronoiWatershedLoss, gwd_sigma_loss)
from mmrotate.models.losses.utils import filter_masks
from mmrotate.registry import MODELS
from projects.point2rbox_v3_multiscale.multiscale_voronoi_loss import (
    MultiScaleVoronoiWatershedLoss)

from .candidate_pool import MultiRadiusWatershedCandidatePool


@MODELS.register_module()
class MultiRadiusPGDMLoss(MultiScaleVoronoiWatershedLoss):
    """PGDM loss with multi-scale SAM and multi-radius watershed pools."""

    def __init__(self,
                 sam_candidate_pool_enabled=False,
                 watershed_candidate_pool_enabled=False,
                 sam_scales=(0.75, 1.0, 1.25),
                 sam_max_masks_per_scale=3,
                 sam_dedup_iou_thr=0.9,
                 watershed_radius_multipliers=(0.75, 1.0, 1.25),
                 watershed_dedup_iou_thr=0.9,
                 **kwargs):
        super().__init__(
            enabled=sam_candidate_pool_enabled,
            scales=sam_scales,
            max_masks_per_scale=sam_max_masks_per_scale,
            dedup_iou_thr=sam_dedup_iou_thr,
            **kwargs)
        self.sam_candidate_pool_enabled = sam_candidate_pool_enabled
        self.watershed_candidate_pool_enabled = \
            watershed_candidate_pool_enabled
        self.watershed_candidate_pool = MultiRadiusWatershedCandidatePool(
            radius_multipliers=watershed_radius_multipliers,
            dedup_iou_thr=watershed_dedup_iou_thr)

    def _forward_multiradius_watershed(self, mu, sigma, label, image,
                                       pos_thres, neg_thres, voronoi):
        candidates_by_instance = self.watershed_candidate_pool.generate(
            image,
            mu,
            sigma,
            label,
            pos_thres,
            neg_thres,
            down_sample=self.down_sample,
            default_sigma=self.default_sigma,
            voronoi=voronoi)
        eigenvalues, eigenvectors = torch.linalg.eigh(sigma)
        instance_losses = []
        height, width = image.shape[-2:]
        markers = torch.full(
            (height, width),
            len(mu) + 1,
            dtype=torch.int32,
            device=mu.device)
        image_min = image.min()
        image_range = image.max() - image_min
        if image_range == 0:
            image_np = torch.zeros_like(image)
        else:
            image_np = (image - image_min) / image_range * 255
        image_np = image_np.permute(
            1, 2, 0).detach().cpu().numpy().astype(np.uint8)

        for instance_index, candidates in enumerate(candidates_by_instance):
            target = eigenvalues[instance_index].detach()
            if candidates:
                masks = [candidate.mask for candidate in candidates]
                scores = np.ones(len(masks), dtype=np.float32)
                class_id = int(label[instance_index].item())
                best_index, _, _ = filter_masks(
                    image,
                    masks,
                    scores,
                    class_id,
                    image_np,
                    mu[instance_index].detach().cpu().numpy(),
                    self.mask_filter_config,
                    self.debug)
                mask_tensor = torch.from_numpy(
                    masks[best_index]).to(mu.device)
                markers[mask_tensor] = instance_index + 1
                coordinates = mask_tensor.nonzero()[:, (1, 0)].float()
                if len(coordinates) > 0:
                    centered = coordinates - mu[instance_index]
                    rotated = eigenvectors[instance_index].T.matmul(
                        centered[:, :, None])[:, :, 0]
                    max_x = torch.max(torch.abs(rotated[:, 0]))
                    max_y = torch.max(torch.abs(rotated[:, 1]))
                    target = torch.stack((max_x, max_y)).square().detach()

            predicted_diagonal = torch.diag_embed(
                eigenvalues[instance_index]).unsqueeze(0)
            target_diagonal = torch.diag_embed(target).unsqueeze(0)
            instance_loss = gwd_sigma_loss(
                predicted_diagonal,
                target_diagonal,
                reduction='none').reshape(-1).mean()
            instance_losses.append(instance_loss)

        losses = torch.stack(instance_losses)
        keep = int(np.ceil(len(losses) * self.topk))
        loss = torch.topk(losses, keep, largest=False)[0].mean()
        self.vis = (markers.clone(), markers)
        return self.loss_weight * loss

    def forward(self,
                pred,
                label,
                image,
                pos_thres,
                neg_thres,
                voronoi='orientation'):
        mu, sigma = pred
        if len(sigma) == 0:
            return self.loss_weight * sigma.sum()
        if len(sigma) <= self.sam_instance_thr:
            if self.sam_candidate_pool_enabled:
                return self._forward_multiscale_sam(mu, sigma, label, image)
            return VoronoiWatershedLoss.forward(
                self, pred, label, image, pos_thres, neg_thres, voronoi)
        if self.watershed_candidate_pool_enabled:
            return self._forward_multiradius_watershed(
                mu, sigma, label, image, pos_thres, neg_thres, voronoi)
        return VoronoiWatershedLoss.forward(
            self, pred, label, image, pos_thres, neg_thres, voronoi)
