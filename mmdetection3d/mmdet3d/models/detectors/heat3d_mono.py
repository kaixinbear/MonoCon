import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from os import path as osp

from mmdet3d.core import (CameraInstance3DBoxes, bbox3d2result,
                          mono_cam_box2vis, show_multi_modality_result)

from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.models.builder import HEADS, build_loss
import torch.nn as nn
from mmdet3d.models.losses import Det2DLoss, DepthLoss
import torch.nn.functional as F

class Feature_2D_to_3D(nn.Module):
    '''
        Transform 2D feature map to 3D format.
        Using auxiliary branch for 2D Detection and depth estimation.
        Args:
            input_channel: input channel of 2D feature map. defalut: 64
            output_channel: channel of semantic feature. default 64
            depth_bins: used for depth loss.
    '''
    def __init__(self, input_channel, output_channel, depth_bins, num_classes):
        super(Feature_2D_to_3D, self).__init__()
        self.semantic_conv = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
            nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU()
        )

        # 2D Detect
        self.sem_heatmap_head = nn.Conv2d(output_channel, num_classes, kernel_size=1)
        self.sem_wh_head = nn.Conv2d(output_channel, 2, kernel_size=1)
        self.sem_offset_head = nn.Conv2d(output_channel, 2, kernel_size=1)

        self.depth_conv = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
            nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU()
        )

        # depth distribution
        self.depth_distribution_head = nn.Sequential(
            # nn.ConvTranspose2d(output_channel, output_channel, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(output_channel),
            # nn.ReLU(),
            nn.ConvTranspose2d(output_channel, depth_bins + 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        channel_dim = 1
        depth_dim = 2

        sem_feat = self.semantic_conv(x)
        depth_feat = self.depth_conv(x)

        center_heatmap_pred = self.sem_heatmap_head(sem_feat).sigmoid()
        center_heatmap_pred = torch.clamp(center_heatmap_pred, min=1e-4, max=1 - 1e-4)
        wh_pred = self.sem_wh_head(sem_feat)
        offset_pred = self.sem_offset_head(sem_feat)

        depth_logits = self.depth_distribution_head(depth_feat)
        depth_probs = F.softmax(depth_logits, dim=depth_dim)

        if self.training:
            aux_outputs = [center_heatmap_pred, wh_pred, offset_pred, depth_probs]
        else:
            aux_outputs = []

        # Resize to match dimensions
        sem_feat = sem_feat.unsqueeze(depth_dim)
        depth_feat = depth_feat.unsqueeze(channel_dim)

        features_3d = sem_feat * depth_feat


        return features_3d, aux_outputs

@DETECTORS.register_module()
class HeatMap_3D_Mono(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 feat_2d_to_3d=None,
                 loss_2d=None,
                 loss_depth_estimation=None
                 ):
        assert bbox_head is not None
        super(HeatMap_3D_Mono, self).__init__(backbone, neck, bbox_head, train_cfg,
                                              test_cfg, pretrained)
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.feat_2d_to_3d = Feature_2D_to_3D(**feat_2d_to_3d)
        self.det_2d_loss = Det2DLoss(**loss_2d)
        self.depth_loss = DepthLoss(**loss_depth_estimation)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      centers2d,
                      depths,
                      gt_kpts_2d=None,
                      gt_kpts_valid_mask=None,
                      attr_labels=None,
                      gt_bboxes_ignore=None,
                      depth_map=None,
                      **kwargs):
        x = self.extract_feat(img)  # 4x downsampling
        x = [self.conv1(x[0])]      # 8x
        feat_shape = x[0].shape
        img_shape = img.shape[2:]
        x, aux_outputs = self.feat_2d_to_3d(x[0])
        losses = {}
        loss_det_2d = self.det_2d_loss(*aux_outputs[:3], gt_bboxes, gt_labels, feat_shape, img_shape, img_metas)
        loss_depth = self.depth_loss(aux_outputs[3], depth_map, gt_bboxes)
        # x = self.heatmap_3d_encoding(x, mode='dense')
        # losses = self.bbox_head.forward_train([x], img_metas, gt_bboxes,
        #                                       gt_labels, gt_bboxes_3d,
        #                                       gt_labels_3d, centers2d, depths,
        #                                       gt_kpts_2d, gt_kpts_valid_mask,
        #                                       attr_labels, gt_bboxes_ignore,
        #                                       **kwargs)
        losses.update(loss_det_2d)
        losses.update(loss_depth)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        x = [self.conv1(x[0])]      # 8x
        feat_shape = x[0].shape
        img_shape = img.shape[2:]
        x, _ = self.feat_2d_to_3d(x[0])
        x = self.heatmap_3d_encoding(x, mode='dense')
        outs = self.bbox_head(x)
        bbox_outputs = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)

        if not self.bbox_head.pred_bbox2d:
            bbox_img = []
            for bbox_output in bbox_outputs:
                bboxes, scores, labels = bbox_output
                bbox_img.append(bbox3d2result(bboxes, scores, labels))
        else:
            from mmdet.core import bbox2result
            bbox2d_img = []
            bbox_img = []
            for bbox_output in bbox_outputs:
                bboxes, scores, labels, bboxes2d = bbox_output
                bbox2d_img.append(bbox2result(bboxes2d, labels, self.bbox_head.num_classes))
                bbox_img.append(bbox3d2result(bboxes, scores, labels))

        bbox_list = [dict() for i in range(len(img_metas))]
        for result_dict, img_bbox in zip(bbox_list, bbox_img):
            result_dict['img_bbox'] = img_bbox
        if self.bbox_head.pred_bbox2d:
            for result_dict, img_bbox2d in zip(bbox_list, bbox2d_img):
                result_dict['img_bbox2d'] = img_bbox2d
        return bbox_list

    def aug_test(self, imgs, img_metas, rescale=True):
        raise NotImplementedError

    def show_results(self, data, result, out_dir):
        raise NotImplementedError

