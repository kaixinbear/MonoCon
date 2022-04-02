import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from os import path as osp
from mmdet.models.builder import HEADS, build_loss, LOSSES
import torch.nn as nn
from mmdet.models.utils.gaussian_target import (gaussian_radius, gen_gaussian_target)
from mmdet.models.utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                                transpose_and_gather_feat)
@LOSSES.register_module()
class Det2DLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 max_objs=30,
                 loss_center_heatmap=None,
                 loss_wh=None,
                 loss_offset=None):
        super(Det2DLoss, self).__init__()
        self.num_classes = num_classes
        self.max_objs = max_objs
        self.loss_center_heatmap = build_loss(loss_center_heatmap)
        self.loss_wh = build_loss(loss_wh)
        self.loss_offset = build_loss(loss_offset)

    @staticmethod
    def extract_input_from_tensor(input, ind, mask):
        input = transpose_and_gather_feat(input, ind)
        return input[mask]

    @staticmethod
    def extract_target_from_tensor(target, mask):
        return target[mask]

    def get_targets(self, gt_bboxes, gt_labels, feat_shape, img_shape, img_metas):
        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        calibs = []

        # objects as 2D center points
        center_heatmap_target = gt_bboxes[-1].new_zeros([bs, self.num_classes, feat_h, feat_w])

        # 2D attributes
        wh_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 2])
        offset_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 2])

        # indices
        indices = gt_bboxes[-1].new_zeros([bs, self.max_objs]).type(torch.cuda.LongTensor)

        # masks
        mask_target = gt_bboxes[-1].new_zeros([bs, self.max_objs])

        for batch_id in range(bs):
            img_meta = img_metas[batch_id]
            cam_p2 = img_meta['cam_intrinsic']

            gt_bbox = gt_bboxes[batch_id]
            calibs.append(cam_p2)
            if len(gt_bbox) < 1:
                continue
            gt_label = gt_labels[batch_id]

            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2
            gt_centers = torch.cat((center_x, center_y), dim=1)

            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio

                radius = gaussian_radius([scale_box_h, scale_box_w],
                                        min_overlap=0.3)
                radius = max(0, int(radius))
                ind = gt_label[j]
                gen_gaussian_target(center_heatmap_target[batch_id, ind],
                                    [ctx_int, cty_int], radius)

                indices[batch_id, j] = cty_int * feat_w + ctx_int

                wh_target[batch_id, j, 0] = scale_box_w
                wh_target[batch_id, j, 1] = scale_box_h
                offset_target[batch_id, j, 0] = ctx - ctx_int
                offset_target[batch_id, j, 1] = cty - cty_int

                mask_target[batch_id, j] = 1

        mask_target = mask_target.type(torch.bool)

        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            wh_target=wh_target,
            offset_target=offset_target,
            indices=indices,
            mask_target=mask_target,
        )

        return target_result

    def forward(self, center_heatmap_pred, wh_pred, offset_pred, gt_bboxes, gt_labels, feat_shape, img_shape, img_metas):
        target_result = self.get_targets(gt_bboxes, gt_labels, feat_shape, img_shape, img_metas)
        center_heatmap_target = target_result['center_heatmap_target']
        wh_target = target_result['wh_target']
        offset_target = target_result['offset_target']
        indices = target_result['indices']
        mask_target = target_result['mask_target']

        # 2d offset
        offset_pred = self.extract_input_from_tensor(offset_pred, indices, mask_target)
        offset_target = self.extract_target_from_tensor(offset_target, mask_target)
        # 2d size
        wh_pred = self.extract_input_from_tensor(wh_pred, indices, mask_target)
        wh_target = self.extract_target_from_tensor(wh_target, mask_target)

        loss_det_2d_center_heatmap = self.loss_center_heatmap(center_heatmap_pred, center_heatmap_target)

        loss_det_2d_wh = self.loss_wh(wh_pred, wh_target)
        loss_det_2d_offset = self.loss_offset(offset_pred, offset_target)

        return dict(
            loss_det_2d_center_heatmap=loss_det_2d_center_heatmap,
            loss_det_2d_wh=loss_det_2d_wh,
            loss_det_2d_offset=loss_det_2d_offset,
        )