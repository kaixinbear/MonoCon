import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from os import path as osp
from mmdet.models.builder import HEADS, build_loss, LOSSES
from mmdet3d.models.utils.gaussian_target_3D import bin_depths
try:
    from kornia.losses.focal import FocalLoss
except:
    from .kornia_focal_loss import FocalLoss

import torch.nn as nn

class Balancer(nn.Module):
    def __init__(self, fg_weight, bg_weight, downsample_factor=1):
        """
        Initialize fixed foreground/background loss balancer
        Args:
            fg_weight: float, Foreground loss weight
            bg_weight: float, Background loss weight
            downsample_factor: int, Depth map downsample factor
        """
        super().__init__()
        self.fg_weight = fg_weight
        self.bg_weight = bg_weight
        self.downsample_factor = downsample_factor

    def compute_fg_mask(self, gt_boxes2d, shape, downsample_factor=1, device=torch.device("cpu")):
        """
        Compute foreground mask for images
        Args:
            gt_boxes2d: [(m1, 4), (m2, 4), ...(mB, 4)], 2D box labels
            shape: torch.Size or tuple, Foreground mask desired shape
            downsample_factor: int, Downsample factor for image
            device: torch.device, Foreground mask desired device
        Returns:
            fg_mask (shape), Foreground mask
        """
        fg_mask = torch.zeros(shape, dtype=torch.bool, device=device)

        for i in range(len(gt_boxes2d)):
            bi_gt_boxes2d = gt_boxes2d[i]
            bi_gt_boxes2d /= downsample_factor
            bi_gt_boxes2d[:, :2] = torch.floor(bi_gt_boxes2d[:, :2])
            bi_gt_boxes2d[:, 2:] = torch.ceil(bi_gt_boxes2d[:, 2:])
            bi_gt_boxes2d = bi_gt_boxes2d.long()
            for n in range(bi_gt_boxes2d.shape[0]):
                u1, v1, u2, v2 = bi_gt_boxes2d[n]
                fg_mask[i, v1:v2, u1:u2] = True

        return fg_mask

    def forward(self, loss, gt_boxes2d):
        """
        Forward pass
        Args:
            loss: (B, H, W), Pixel-wise loss
            gt_boxes2d: [(m1, 4), (m2, 4), ...(mB, 4)], 2D box labels for foreground/background balancing
        Returns:
            loss: (1), Total loss after foreground/background balancing
            tb_dict: dict[float], All losses to log in tensorboard
        """
        # Compute masks
        fg_mask = self.compute_fg_mask(gt_boxes2d=gt_boxes2d,
                                             shape=loss.shape,
                                             downsample_factor=self.downsample_factor,
                                             device=loss.device)
        bg_mask = ~fg_mask

        # Compute balancing weights
        weights = self.fg_weight * fg_mask + self.bg_weight * bg_mask
        num_pixels = fg_mask.sum() + bg_mask.sum()

        # Compute losses
        loss *= weights
        fg_loss = loss[fg_mask].sum() / num_pixels
        bg_loss = loss[bg_mask].sum() / num_pixels

        # Get total loss
        loss = fg_loss + bg_loss
        tb_dict = {"fg_loss": fg_loss, "bg_loss": bg_loss}
        return loss, tb_dict


@LOSSES.register_module()
class DepthLoss(nn.Module):
    def __init__(self,
                 weight,
                 alpha,
                 gamma,
                 disc_cfg,
                 fg_weight,
                 bg_weight,
                 downsample_factor):
        """
        Initializes DDNLoss module
        Args:
            weight: float, Loss function weight
            alpha: float, Alpha value for Focal Loss
            gamma: float, Gamma value for Focal Loss
            disc_cfg: dict, Depth discretiziation configuration
            fg_weight: float, Foreground loss weight
            bg_weight: float, Background loss weight
            downsample_factor: int, Depth map downsample factor
        """
        super().__init__()
        self.device = torch.cuda.current_device()
        self.disc_cfg = disc_cfg
        self.balancer = Balancer(downsample_factor=downsample_factor,
                                 fg_weight=fg_weight,
                                 bg_weight=bg_weight)

        # Set loss function
        self.alpha = alpha
        self.gamma = gamma
        self.loss_func = FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction="none")
        self.weight = weight

    def forward(self, depth_logits, depth_maps, gt_boxes2d):
        """
        Gets DDN loss
        Args:
            depth_logits: (B, D+1, H, W), Predicted depth logits
            depth_maps: (B, H, W), Depth map [m]
            gt_boxes2d: torch.Tensor (B, N, 4), 2D box labels for foreground/background balancing
        Returns:
            loss: (1), Depth distribution network loss
            tb_dict: dict[float], All losses to log in tensorboard
        """
        tb_dict = {}

        # Bin depth map to create target
        depth_target = bin_depths(depth_maps, **self.disc_cfg, target=True)

        # Compute loss
        loss = self.loss_func(depth_logits, depth_target)

        # Compute foreground/background balancing
        loss, tb_dict = self.balancer(loss=loss, gt_boxes2d=gt_boxes2d)

        # Final loss
        loss *= self.weight

        tb_dict.update({"loss_depth_estimate": loss})
        return tb_dict

