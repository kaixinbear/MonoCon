import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from os import path as osp
from mmcv.ops import batched_nms
from mmdet3d.core import (CameraInstance3DBoxes, bbox3d2result,
                          mono_cam_box2vis, show_multi_modality_result)

from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                                transpose_and_gather_feat)
import torch.nn as nn
from mmdet3d.models.losses import Det2DLoss, DepthLoss
import torch.nn.functional as F
from mmcv.runner import force_fp32
from .sparse_backbone3d import SparseBackbone3D

class Feature_2D_to_3D(nn.Module):
    '''
        Transform 2D feature map to 3D format.
        Using auxiliary branch for 2D Detection and depth estimation.
        Args:
            input_channel: input channel of 2D feature map. defalut: 64
            output_channel: channel of semantic feature. default 64
            depth_bins: used for depth loss.
    '''
    def __init__(self, test_cfg, input_channel, output_channel, depth_bins, num_classes, ):
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
        # self.depth_distribution_head = nn.Sequential(
        #     # nn.ConvTranspose2d(output_channel, output_channel, kernel_size=4, stride=2, padding=1),
        #     # nn.BatchNorm2d(output_channel),
        #     # nn.ReLU(),
        #     nn.ConvTranspose2d(output_channel, depth_bins + 1, kernel_size=4, stride=2, padding=1),
        # )
        self.depth_distribution_head = nn.Conv2d(output_channel, depth_bins + 1, kernel_size=3, stride=1, padding=1)

        self.test_cfg = test_cfg

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
        aux_outputs = [center_heatmap_pred, wh_pred, offset_pred, depth_logits]

        # Resize to match dimensions
        sem_feat = sem_feat.unsqueeze(depth_dim)
        depth_logits = depth_logits.unsqueeze(channel_dim)
        depth_probs = F.softmax(depth_logits, dim=depth_dim)
        depth_probs = depth_probs[:, :, :-1]

        features_3d = sem_feat * depth_probs

        return features_3d, aux_outputs

    @force_fp32(apply_to=('center_heatmap_preds', 'wh_preds', 'offset_preds'))
    def get_bboxes(self,
                   center_heatmap_preds,
                   wh_preds,
                   offset_preds,
                   img_metas,
                   rescale=True,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            center_heatmap_preds (Tensor): Center predict heatmaps for
                all levels with shape (B, num_classes, H, W).
            wh_preds (Tensor): WH predicts for all levels with
                shape (B, 2, H, W).
            offset_preds (Tensor): Offset predicts for all levels
                with shape (B, 2, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: True.
            with_nms (bool): If True, do nms before return boxes.
                Default: False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        result_list = []
        # print(self.test_cfg)
        for img_id in range(len(img_metas)):
            result_list.append(
                self._get_bboxes_single(
                    center_heatmap_preds[img_id:img_id + 1, ...],
                    wh_preds[img_id:img_id + 1, ...],
                    offset_preds[img_id:img_id + 1, ...],
                    img_metas[img_id],
                    rescale=rescale,
                    with_nms=with_nms))
        return result_list

    def _get_bboxes_single(self,
                           center_heatmap_pred,
                           wh_pred,
                           offset_pred,
                           img_meta,
                           rescale=False,
                           with_nms=True):
        """Transform outputs of a single image into bbox results.

        Args:
            center_heatmap_pred (Tensor): Center heatmap for current level with
                shape (1, num_classes, H, W).
            wh_pred (Tensor): WH heatmap for current level with shape
                (1, num_classes, H, W).
            offset_pred (Tensor): Offset for current level with shape
                (1, corner_offset_channels, H, W).
            img_meta (dict): Meta information of current image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor, Tensor]: The first item is an (n, 5) tensor, where
                5 represent (tl_x, tl_y, br_x, br_y, score) and the score
                between 0 and 1. The shape of the second tensor in the tuple
                is (n,), and each element represents the class label of the
                corresponding box.
        """
        if 'batch_input_shape' not in img_meta.keys():
            img_meta['batch_input_shape'] = img_meta['pad_shape'][:2]
        if 'scale_factor' not in img_meta.keys():
            img_meta['scale_factor'] = center_heatmap_pred.shape[2] / img_meta['batch_input_shape'][0]
            assert img_meta['scale_factor'] == 1/4, print("img_meta['scale_factor']", img_meta['scale_factor'])
        batch_det_bboxes, batch_labels = self.decode_heatmap(
            center_heatmap_pred,
            wh_pred,
            offset_pred,
            img_meta['batch_input_shape'],
            k=self.test_cfg.topk,
            kernel=self.test_cfg.local_maximum_kernel)

        det_bboxes = batch_det_bboxes.view([-1, 5])
        det_labels = batch_labels.view(-1)

        if rescale:
            det_bboxes[..., :4] /= det_bboxes.new_tensor(
                img_meta['scale_factor'])

        if with_nms:
            det_bboxes, det_labels = self._bboxes_nms(det_bboxes, det_labels,
                                                      self.test_cfg)
        return det_bboxes, det_labels

    def decode_heatmap(self,
                       center_heatmap_pred,
                       wh_pred,
                       offset_pred,
                       img_shape,
                       k=100,
                       kernel=3):
        """Transform outputs into detections raw bbox prediction.

        Args:
            center_heatmap_pred (Tensor): center predict heatmap,
               shape (B, num_classes, H, W).
            wh_pred (Tensor): wh predict, shape (B, 2, H, W).
            offset_pred (Tensor): offset predict, shape (B, 2, H, W).
            img_shape (list[int]): image shape in [h, w] format.
            k (int): Get top k center keypoints from heatmap. Default 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
               Default 3.

        Returns:
            tuple[torch.Tensor]: Decoded output of CenterNetHead, containing
               the following Tensors:

              - batch_bboxes (Tensor): Coords of each box with shape (B, k, 5)
              - batch_topk_labels (Tensor): Categories of each box with \
                  shape (B, k)
        """
        height, width = center_heatmap_pred.shape[2:]
        inp_h, inp_w = img_shape

        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        wh = transpose_and_gather_feat(wh_pred, batch_index)
        offset = transpose_and_gather_feat(offset_pred, batch_index)
        topk_xs = topk_xs + offset[..., 0]
        topk_ys = topk_ys + offset[..., 1]
        tl_x = (topk_xs - wh[..., 0] / 2) * (inp_w / width)
        tl_y = (topk_ys - wh[..., 1] / 2) * (inp_h / height)
        br_x = (topk_xs + wh[..., 0] / 2) * (inp_w / width)
        br_y = (topk_ys + wh[..., 1] / 2) * (inp_h / height)

        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]),
                                 dim=-1)
        return batch_bboxes, batch_topk_labels

    def _bboxes_nms(self, bboxes, labels, cfg):
        if labels.numel() > 0:
            max_num = cfg.max_per_img
            bboxes, keep = batched_nms(bboxes[:, :4], bboxes[:,
                                                             -1].contiguous(),
                                       labels, cfg.nms)
            if max_num > 0:
                bboxes = bboxes[:max_num]
                labels = labels[keep][:max_num]

        return bboxes, labels


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
                 downsample_rate_on_4x=None,
                 loss_2d=None,
                 loss_depth_estimation=None,
                 heat3d_encoding=None,
                 ):
        assert bbox_head is not None
        super(HeatMap_3D_Mono, self).__init__(backbone, neck, bbox_head, train_cfg,
                                              test_cfg, pretrained)
        self.downsample_factor = loss_depth_estimation.downsample_factor
        self.feat_2d_to_3d = Feature_2D_to_3D(test_cfg, **feat_2d_to_3d, )
        self.det_2d_loss = Det2DLoss(**loss_2d)
        self.depth_loss = DepthLoss(**loss_depth_estimation)

        self.heatmap_3d_encoding = SparseBackbone3D(**heat3d_encoding)

        if downsample_rate_on_4x:
            self.downsample_conv = nn.ModuleList([])
            for _ in range(downsample_rate_on_4x//2):
                self.downsample_conv.append(nn.Conv2d(feat_2d_to_3d.input_channel, feat_2d_to_3d.input_channel, kernel_size=3, stride=2, padding=1))
        else:
            self.downsample_conv = None

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
        if self.downsample_conv:
            for i in range(len(self.downsample_conv)):
                x[0] = self.downsample_conv[i](x[0])
        feat_shape = x[0].shape
        img_shape = img.shape[2:]
        x, aux_outputs = self.feat_2d_to_3d(x[0])
        losses = {}
        loss_det_2d = self.det_2d_loss(*aux_outputs[:3], gt_bboxes, gt_labels, feat_shape, img_shape, img_metas)
        loss_depth, gt_voxel_mask = self.depth_loss(aux_outputs[3], depth_map, gt_bboxes)

        ##　测试的时候是没有GT 2D BBox的，所以需用用CenterNet的2D检测结果来当作GT;
        # center_heatmap_pred, wh_pred, offset_pred, _ = aux_outputs
        # det_2d_result_list = self.feat_2d_to_3d.get_bboxes(center_heatmap_pred, wh_pred, offset_pred, img_metas)
        # pred_fg_mask = self.generate_mask_via_2DBox(det_2d_result_list, gt_fg_mask.shape)

        # input_dict = self.prepare_for_heat3d(x, pred_fg_mask)

        ## use gt foreground mask as sparse conv indices.
        input_dict = self.prepare_for_heat3d(x, gt_voxel_mask)

        x = self.heatmap_3d_encoding(input_dict)
        loss_3d = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_3d,
                                              gt_labels_3d, centers2d, depths,
                                              gt_kpts_2d, gt_kpts_valid_mask,
                                              attr_labels, gt_bboxes_ignore,
                                              **kwargs)

        losses.update(loss_det_2d)
        losses.update(loss_depth)
        losses.update(loss_3d)
        return losses

    def prepare_for_heat3d(self, feat_3d, gt_voxel_mask):
        '''
            Args: feat_3d: (B, C, D, H, W)
                  fg_2d_mask: (B, H, W)
        '''
        input_dict = {}
        B, C, D, H, W = feat_3d.shape
        feat_3d = feat_3d.permute(0, 3, 4, 2, 1).contiguous() #(B, H, W, D, C)
        # print("gt_voxel_mask", gt_voxel_mask.sum())
        feat_masked_3d = feat_3d[gt_voxel_mask] # your features with shape [N, num_channels]
        indices = torch.nonzero(gt_voxel_mask).int().contiguous() # your indices/coordinates with shape [N, ndim + 1], batch index must be put in indices[:, 0]
        
        input_dict['sparse_shape'] = [H, W, D]
        input_dict['voxel_features'] = feat_masked_3d
        input_dict['voxel_indices'] = indices
        input_dict['batch_size'] = B

        return input_dict

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        # x = [self.conv1(x[0])]      # 8x
        feat_shape = x[0].shape
        img_shape = img.shape[2:]
        x, aux_outputs = self.feat_2d_to_3d(x[0])
        center_heatmap_pred, wh_pred, offset_pred, _ = aux_outputs
        det_2d_result_list = self.feat_2d_to_3d.get_bboxes(center_heatmap_pred, wh_pred, offset_pred, img_metas)
        
        ## eval 2D Detection Only
        # from mmdet.core import bbox2result
        # bbox2d_img = []
        # bbox_img = []
        # for bbox_output in det_2d_result_list:
        #     bboxes2d, labels = bbox_output
        #     score_mask = bboxes2d[..., -1] > 0.01
        #     bboxes2d, labels = bboxes2d[score_mask], labels[score_mask]
        #     bbox2d_img.append(bbox2result(bboxes2d, labels, self.bbox_head.num_classes))

        # bbox_list = [dict() for i in range(len(img_metas))]
        # for result_dict, img_bbox2d in zip(bbox_list, bbox2d_img):
        #     result_dict['img_bbox2d'] = img_bbox2d
        # return bbox_list

        ## eval 3D Dtection
        B, _, feat_H, feat_W = center_heatmap_pred.shape
        pred_fg_mask = self.generate_mask_via_2DBox(det_2d_result_list, (B, feat_H, feat_W))
        input_dict = self.prepare_for_heat3d(x, pred_fg_mask)
        x = self.heatmap_3d_encoding(input_dict)
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

    def generate_mask_via_2DBox(self, det_2d_result_list, mask_shape):
        pred_fg_mask = torch.zeros(mask_shape)
        for batch_id, bbox_output in enumerate(det_2d_result_list):
            bboxes2d, labels = bbox_output
            score_mask = bboxes2d[..., -1] > 0.01
            bboxes2d, labels = bboxes2d[score_mask], labels[score_mask]
            print("bboxes", bboxes2d, bboxes2d.shape)
            if bboxes2d.shape[0] > 0:
                bboxes2d = bboxes2d[:, :4]
                bboxes2d /= self.downsample_factor
                bboxes2d[:, :2] = torch.floor(bboxes2d[:, :2])
                bboxes2d[:, 2:] = torch.ceil(bboxes2d[:, 2:])
                bboxes2d = bboxes2d.long()
                for n in range(bboxes2d.shape[0]):
                    u1, v1, u2, v2 = bboxes2d[n]
                    pred_fg_mask[batch_id, v1:v2, u1:u2] = True
        return pred_fg_mask

    def aug_test(self, imgs, img_metas, rescale=True):
        raise NotImplementedError

    def show_results(self, data, result, out_dir):
        raise NotImplementedError