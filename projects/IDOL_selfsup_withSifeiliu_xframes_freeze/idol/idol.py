# ------------------------------------------------------------------------
# IDOL: In Defense of Online Models for Video Instance Segmentation
# Copyright (c) 2022 ByteDance. All Rights Reserved.
# ------------------------------------------------------------------------

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from typing import Dict, List
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
####################################################################################################
#boxInst 引用
from detectron2.structures import ImageList
from skimage import color
####################################################################################################
from fvcore.nn import giou_loss, smooth_l1_loss
from .models.backbone import Joiner
from .models.deformable_detr import DeformableDETR, SetCriterion
from .models.matcher import HungarianMatcher
from .models.position_encoding import PositionEmbeddingSine
from .models.deformable_transformer import DeformableTransformer
from .models.segmentation_condInst import CondInst_segm, segmentation_postprocess
from .models.tracker import IDOL_Tracker
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.misc import NestedTensor
from .data.coco import convert_coco_poly_to_mask
import torchvision.ops as ops


INF = 100000000
from detectron2.layers import cat

__all__ = ["IDOL"]

class MaskedBackbone(nn.Module):
    """ This is a thin wrapper around D2's backbone to provide padding masking"""

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        #######################################################
        self.backbone_shape = backbone_shape
        #######################################################
        self.feature_strides = [backbone_shape[f].stride for f in backbone_shape.keys()]
        self.num_channels = [backbone_shape[f].channels for f in backbone_shape.keys()]
        
    def forward(self, tensor_list):
        xs = self.backbone(tensor_list.tensors)
        # xs = self.backbone(tensor_list.tensor)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

    def mask_out_padding(self, feature_shapes, image_sizes, device):
        masks = []
        assert len(feature_shapes) == len(self.feature_strides)
        for idx, shape in enumerate(feature_shapes):
            N, _, H, W = shape
            masks_per_feature_level = torch.ones((N, H, W), dtype=torch.bool, device=device)
            for img_idx, (h, w) in enumerate(image_sizes):
                masks_per_feature_level[
                    img_idx,
                    : int(np.ceil(float(h) / self.feature_strides[idx])),
                    : int(np.ceil(float(w) / self.feature_strides[idx])),
                ] = 0
            masks.append(masks_per_feature_level)
        return masks


@META_ARCH_REGISTRY.register()
class IDOL(nn.Module):
    """
    Implement IDOL
    """
    def __init__(self, cfg):
        super().__init__()
        self.num_frames = cfg.INPUT.SAMPLING_FRAME_NUM

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.clip_stride = cfg.MODEL.IDOL.CLIP_STRIDE

        ### inference setting
        self.merge_on_cpu = cfg.MODEL.IDOL.MERGE_ON_CPU
        self.is_multi_cls = cfg.MODEL.IDOL.MULTI_CLS_ON
        self.apply_cls_thres = cfg.MODEL.IDOL.APPLY_CLS_THRES
        self.temporal_score_type = cfg.MODEL.IDOL.TEMPORAL_SCORE_TYPE
        self.inference_select_thres = cfg.MODEL.IDOL.INFERENCE_SELECT_THRES
        self.inference_fw = cfg.MODEL.IDOL.INFERENCE_FW
        self.inference_tw = cfg.MODEL.IDOL.INFERENCE_TW
        self.memory_len = cfg.MODEL.IDOL.MEMORY_LEN
        self.nms_pre = cfg.MODEL.IDOL.NMS_PRE
        self.add_new_score = cfg.MODEL.IDOL.ADD_NEW_SCORE 
        self.batch_infer_len = cfg.MODEL.IDOL.BATCH_INFER_LEN


        self.is_coco = cfg.DATASETS.TEST[0].startswith("coco")
        self.num_classes = cfg.MODEL.IDOL.NUM_CLASSES
        self.mask_stride = cfg.MODEL.IDOL.MASK_STRIDE
        self.match_stride = cfg.MODEL.IDOL.MATCH_STRIDE
        self.mask_on = cfg.MODEL.MASK_ON

        self.coco_pretrain = cfg.INPUT.COCO_PRETRAIN
        hidden_dim = cfg.MODEL.IDOL.HIDDEN_DIM
        num_queries = cfg.MODEL.IDOL.NUM_OBJECT_QUERIES

        # Transformer parameters:
        nheads = cfg.MODEL.IDOL.NHEADS
        dropout = cfg.MODEL.IDOL.DROPOUT
        dim_feedforward = cfg.MODEL.IDOL.DIM_FEEDFORWARD
        enc_layers = cfg.MODEL.IDOL.ENC_LAYERS
        dec_layers = cfg.MODEL.IDOL.DEC_LAYERS
        enc_n_points = cfg.MODEL.IDOL.ENC_N_POINTS
        dec_n_points = cfg.MODEL.IDOL.DEC_N_POINTS
        num_feature_levels = cfg.MODEL.IDOL.NUM_FEATURE_LEVELS

        # Loss parameters:
        mask_weight = cfg.MODEL.IDOL.MASK_WEIGHT
        dice_weight = cfg.MODEL.IDOL.DICE_WEIGHT
        giou_weight = cfg.MODEL.IDOL.GIOU_WEIGHT
        l1_weight = cfg.MODEL.IDOL.L1_WEIGHT
        class_weight = cfg.MODEL.IDOL.CLASS_WEIGHT
        reid_weight = cfg.MODEL.IDOL.REID_WEIGHT
        deep_supervision = cfg.MODEL.IDOL.DEEP_SUPERVISION

        focal_alpha = cfg.MODEL.IDOL.FOCAL_ALPHA

        set_cost_class = cfg.MODEL.IDOL.SET_COST_CLASS
        set_cost_bbox = cfg.MODEL.IDOL.SET_COST_BOX
        set_cost_giou = cfg.MODEL.IDOL.SET_COST_GIOU

        N_steps = hidden_dim // 2
        d2_backbone = MaskedBackbone(cfg)
        backbone = Joiner(d2_backbone, PositionEmbeddingSine(N_steps, normalize=True))
        backbone.num_channels = d2_backbone.num_channels[1:]  # only take [c3 c4 c5] from resnet and gengrate c6 later
        backbone.strides = d2_backbone.feature_strides[1:]
        ################################################
        #### FCOS
        # backbone.backbone_shape = {}
        # count_flag = 0
        # for key,item in d2_backbone.backbone_shape.items():
        #     count_flag+=1
        #     if count_flag>1:
        #         backbone.backbone_shape[key] = item
                
        # self.proposal_generator = build_proposal_generator(cfg, backbone.backbone_shape)

        # in_channels = self.proposal_generator.in_channels_to_top_module
        # self.controller = nn.Conv2d(
        #     in_channels, 640,
        #     kernel_size=3, stride=1, padding=1
        # )
        # torch.nn.init.normal_(self.controller.weight, std=0.01)
        # torch.nn.init.constant_(self.controller.bias, 0)

        # self.backbone = backbone
        ################################################
        
        transformer = DeformableTransformer(
        d_model= hidden_dim,
        nhead=nheads,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_frames=self.num_frames,
        num_feature_levels=num_feature_levels,
        dec_n_points=dec_n_points,
        enc_n_points=enc_n_points,)
        
        
        model = DeformableDETR(
        backbone,
        transformer,
        num_classes=self.num_classes,
        num_frames=self.num_frames,
        num_queries=num_queries,
        num_feature_levels=num_feature_levels,
        aux_loss=deep_supervision,
        with_box_refine=True )

        self.detr = CondInst_segm(model, freeze_detr=False, rel_coord=True )
        
        self.detr.to(self.device)

        # building criterion
        matcher = HungarianMatcher(multi_frame=True, # True, False
                            cost_class=set_cost_class,
                            cost_bbox=set_cost_bbox,
                            cost_giou=set_cost_giou)

        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou":giou_weight}
        weight_dict["loss_reid"] = reid_weight
        weight_dict["loss_reid_aux"] = reid_weight*1.5
        # weight_dict["loss_mask"] = mask_weight
        # weight_dict["loss_dice"] = dice_weight
        ############################################################
        weight_dict["loss_prj"] = mask_weight
        weight_dict["loss_pairwise"] = mask_weight
        # weight_dict["loss_boxinst"] = mask_weight
        ############################################################

        ##################################################################
        weight_dict["loss_selfsup_walk"] = dice_weight
        ##################################################################
        if deep_supervision:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
  
        # losses = ['labels', 'boxes', 'masks','reid']
        ######################################################################
        # losses = ['labels', 'boxes', 'masks','reid', 'loss_prj','loss_pairwise']
        # losses = ['labels', 'boxes', 'masks','reid', 'loss_boxinst']

        # losses = ['labels', 'boxes', 'masks','reid', 'no_masks']

        # losses = ['labels', 'boxes', 'reid', 'no_masks']
        ######################################################################

        ##################################################################
        # losses = ['labels', 'boxes', 'reid', 'no_masks','selfsup_walk']
        losses = ['selfsup_walk']
        ##################################################################
        

        self.criterion = SetCriterion(self.num_classes, matcher, weight_dict, losses, 
                             mask_out_stride=self.mask_stride,
                             focal_alpha=focal_alpha,
                             num_frames = self.num_frames)
        self.criterion.to(self.device)
        
        self.criterion.pairwise_color_thresh = cfg.MODEL.Boxinst.pairwise_color_thresh
        self.criterion._warmup_iters = cfg.MODEL.Boxinst._warmup_iters

        self.pairwise_dilation = cfg.MODEL.Boxinst.PAIRWIAE_DILATTION
        self.pairwise_size = cfg.MODEL.Boxinst.PAIRWISE_SIZE

        self.criterion.pairwise_dilation = cfg.MODEL.Boxinst.PAIRWIAE_DILATTION
        self.criterion.pairwise_size = cfg.MODEL.Boxinst.PAIRWISE_SIZE

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

        self.merge_device = "cpu" if self.merge_on_cpu else self.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances: Instances
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        if self.training:    
            ######################################################
            self.criterion.batched_inputs = batched_inputs
            ######################################################
            
            # 关于BoxInst的相关内容区域
            gt_instances=[]
            original_images=[]
            for x in batched_inputs:
                for itemm in x["instances"]:
                    gt_instances.append(itemm.to(self.device))
                for xitem in x["image"]:
                    original_images.append(xitem.to(self.device))

            ###############################################################################
            # original_image_masks = [torch.ones_like(x[0], dtype=torch.float32) for x in original_images]
            # # mask out the bottom area where the COCO dataset probably has wrong annotations
            # for i in range(len(original_image_masks)):
            #     im_h = batched_inputs[int(i/self.num_frames)]["height"]
            #     # im_h = batched_inputs[i]["height"]
            #     # self.bottom_pixels_removed = 10
            #     pixels_removed = int(
            #         10 *
            #         float(original_images[i].size(1)) / float(im_h)
            #     )
            #     if pixels_removed > 0:
            #         original_image_masks[i][-pixels_removed:, :] = 0
            # # self.backbone.size_divisibility=32
            # original_images = ImageList.from_tensors(original_images, 32)
            # original_image_masks = ImageList.from_tensors(
            #     original_image_masks, 32, pad_value=0.0
            # )
            # self.add_bitmasks_from_boxes(
            #     gt_instances, original_images.tensor, original_image_masks.tensor,
            #     original_images.tensor.size(-2), original_images.tensor.size(-1)
            # )        
            ###############################################################################

            ############################################################
            # bz = len(gt_instances)//2
            # key_ids = list(range(0,bz*2-1,2))
            # ref_ids = list(range(1,bz*2,2))

            # det_original_images = [original_images[_i] for _i in key_ids]
            # ref_original_images = [original_images[_i] for _i in ref_ids]

            # det_gt_instances = [gt_instances[_i] for _i in key_ids]
            # ref_gt_instances = [gt_instances[_i] for _i in ref_ids]
            # ############################################################
            # # if self.boxinst_enabled:
            # det_original_image_masks = [torch.ones_like(x[0], dtype=torch.float32) for x in det_original_images]
            # # mask out the bottom area where the COCO dataset probably has wrong annotations
            # for i in range(len(det_original_image_masks)):
            #     im_h = batched_inputs[int(i/2)]["height"]
            #     # im_h = batched_inputs[i]["height"]
            #     # self.bottom_pixels_removed = 10
            #     pixels_removed = int(
            #         10 *
            #         float(det_original_images[i].size(1)) / float(im_h)
            #     )
            #     if pixels_removed > 0:
            #         det_original_image_masks[i][-pixels_removed:, :] = 0
            # # self.backbone.size_divisibility=32
            # det_original_images = ImageList.from_tensors(det_original_images, 32)
            # det_original_image_masks = ImageList.from_tensors(
            #     det_original_image_masks, 32, pad_value=0.0
            # )
            # #################################
            # # 如果这里只改成使用关键帧
            # #################################
            # self.add_bitmasks_from_boxes(
            #     det_gt_instances, det_original_images.tensor, det_original_image_masks.tensor,
            #     det_original_images.tensor.size(-2), det_original_images.tensor.size(-1)
            # )
            ################################################################################################################################
        if self.training: 
            images = self.preprocess_image(batched_inputs)
            ######################################################
            # from .util.misc import NestedTensor, interpolate, nested_tensor_from_tensor_list, inverse_sigmoid
            # samples_images = images
            # if not isinstance(samples_images, NestedTensor):
            #     samples_images = nested_tensor_from_tensor_list(samples_images, size_divisibility=32)
            # features,pos = self.backbone(samples_images)
            
            # # proposals, proposal_losses = self.proposal_generator(
            # #                 images, features, gt_instances, self.controller
            # #             )
            # #计算locations
            # locations = self.compute_locations(features)
            # # 计算target_inds
            # training_targets = self._get_ground_truth(locations, gt_instances)

            # # Collect all logits and regression predictions over feature maps
            # # and images to arrive at the same shape as the labels and targets
            # # The final ordering is L, N, H, W from slowest to fastest axis.

            # instances = Instances((0, 0))
            # instances.labels = cat([
            #     # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
            #     x.reshape(-1) for x in training_targets["labels"]
            # ], dim=0)
            # instances.gt_inds = cat([
            #     # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
            #     x.reshape(-1) for x in training_targets["target_inds"]
            # ], dim=0)
            # instances.im_inds = cat([
            #     x.reshape(-1) for x in training_targets["im_inds"]
            # ], dim=0)
            # instances.reg_targets = cat([
            #     # Reshape: (N, Hi, Wi, 4) -> (N*Hi*Wi, 4)
            #     x.reshape(-1, 4) for x in training_targets["reg_targets"]
            # ], dim=0,)
            # instances.locations = cat([
            #     x.reshape(-1, 2) for x in training_targets["locations"]
            # ], dim=0)
            # # instances.fpn_levels = cat([
            # #     x.reshape(-1) for x in training_targets["fpn_levels"]
            # # ], dim=0)

            # # instances.logits_pred = cat([
            # #     # Reshape: (N, C, Hi, Wi) -> (N, Hi, Wi, C) -> (N*Hi*Wi, C)
            # #     x.permute(0, 2, 3, 1).reshape(-1, self.num_classes) for x in logits_pred
            # # ], dim=0,)

            # # instances.reg_pred = cat([
            # #     # Reshape: (N, B, Hi, Wi) -> (N, Hi, Wi, B) -> (N*Hi*Wi, B)
            # #     x.permute(0, 2, 3, 1).reshape(-1, 4) for x in reg_pred
            # # ], dim=0,)
            # # instances.ctrness_pred = cat([
            # #     # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
            # #     x.permute(0, 2, 3, 1).reshape(-1) for x in ctrness_pred
            # # ], dim=0,)
            # # if len(top_feats) > 0:
            # #     instances.top_feats = cat([
            # #         # Reshape: (N, -1, Hi, Wi) -> (N*Hi*Wi, -1)
            # #         x.permute(0, 2, 3, 1).reshape(-1, x.size(1)) for x in top_feats
            # #     ], dim=0,)
            # #获取gt_inds
            # self.criterion.instance_of_fcos = instances

            ######################################################
            det_targets,ref_targets = self.prepare_targets(gt_instances) 
            #gt_instances 与 images 数据相同，格式不同 //det_targets,ref_targets 关键帧和参考帧
            # output, loss_dict = self.detr(images, det_targets,ref_targets, self.criterion, train=True)
            ##############################################################################################################
            output, loss_dict = self.detr(images, det_targets,ref_targets, self.criterion, train=True, mask_feat_stride=8, gt_instances=gt_instances)
            #mask_feats  pred_instances 这里给不了
            # output, loss_dict = self.detr(images, det_targets,ref_targets, self.criterion, train=True, mask_feats, mask_feat_stride=8, pred_instances, gt_instances=gt_instances)
            ##############################################################################################################
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            ####################################################
            # loss_dict.update(proposal_losses)
            ####################################################
            return loss_dict
        elif self.coco_pretrain:  #evluate during coco pretrain
            images = self.preprocess_coco_image(batched_inputs)
            output = self.detr.inference_forward(images, size_divisib=32) #
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            mask_pred = output["pred_masks"] if self.mask_on else None
            results = self.coco_inference(box_cls, box_pred, mask_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = segmentation_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        else:
            images = self.preprocess_image(batched_inputs)
            video_len = len(batched_inputs[0]['file_names'])
            clip_length = self.batch_infer_len
            #split long video into clips to form a batch input 
            if video_len > clip_length:
                num_clips = math.ceil(video_len/clip_length)
                logits_list, boxes_list, embed_list, points_list, masks_list = [], [], [], [], []
                for c in range(num_clips):
                    start_idx = c*clip_length
                    end_idx = (c+1)*clip_length
                    clip_inputs = [{'image':batched_inputs[0]['image'][start_idx:end_idx]}]
                    clip_images = self.preprocess_image(clip_inputs)
                    clip_output = self.detr.inference_forward(clip_images)
                    logits_list.append(clip_output['pred_logits'])
                    boxes_list.append(clip_output['pred_boxes'])
                    embed_list.append(clip_output['pred_inst_embed'])
                    # points_list.append(clip_output['reference_points'])
                    masks_list.append(clip_output['pred_masks'].to(self.merge_device))
                output = {
                    'pred_logits':torch.cat(logits_list,dim=0),
                    'pred_boxes':torch.cat(boxes_list,dim=0),
                    'pred_inst_embed':torch.cat(embed_list,dim=0),
                    # 'reference_points':torch.cat(points_list,dim=0),
                    'pred_masks':torch.cat(masks_list,dim=0),
                }    
            else:
                images = self.preprocess_image(batched_inputs)
                output = self.detr.inference_forward(images)
            idol_tracker = IDOL_Tracker(
                    init_score_thr= 0.2,
                    obj_score_thr=0.1,
                    nms_thr_pre=self.nms_pre,  #0.5
                    nms_thr_post=0.05,
                    addnew_score_thr = self.add_new_score, #0.2
                    memo_tracklet_frames = 10,
                    memo_momentum = 0.8,
                    long_match = self.inference_tw,
                    frame_weight = (self.inference_tw|self.inference_fw),
                    temporal_weight = self.inference_tw,
                    memory_len = self.memory_len
                    )
            height = batched_inputs[0]['height']
            width = batched_inputs[0]['width']
            video_output = self.inference(output, idol_tracker, (height, width), images.image_sizes[0])  # (height, width) is resized size,images. image_sizes[0] is original size

            return video_output

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            ############################################################
            #去除gt_masks
            ############################################################
            # gt_masks = targets_per_image.gt_masks.tensor
            inst_ids = targets_per_image.gt_ids
            valid_id = inst_ids!=-1  # if a object is disappeared，its gt_ids is -1
            # new_targets.append({"labels": gt_classes, "boxes": gt_boxes, 'masks': gt_masks, 'inst_id':inst_ids, 'valid':valid_id})
            ############################################################
            #去除gt_masks
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes, 'inst_id':inst_ids, 'valid':valid_id})
            ############################################################
        bz = len(new_targets)//2
        key_ids = list(range(0,bz*2-1,2))
        ref_ids = list(range(1,bz*2,2))
        det_targets = [new_targets[_i] for _i in key_ids]
        ref_targets = [new_targets[_i] for _i in ref_ids]
        for i in range(bz):  # fliter empety object in key frame
            det_target = det_targets[i]
            ref_target = ref_targets[i]
            if False in det_target['valid']:
                valid_i = det_target['valid'].clone()
                for k,v in det_target.items():
                    det_target[k] = v[valid_i]
                for k,v in ref_target.items():
                    ref_target[k] = v[valid_i]
        return det_targets,ref_targets

    def inference(self, outputs, tracker, ori_size, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        # results = []
        video_dict = {}
        vido_logits = outputs['pred_logits']
        video_output_masks = outputs['pred_masks']
        output_h, output_w = video_output_masks.shape[-2:]
        video_output_boxes = outputs['pred_boxes']
        video_output_embeds = outputs['pred_inst_embed']
        vid_len = len(vido_logits)
        for i_frame, (logits, output_mask, output_boxes, output_embed) in enumerate(zip(
            vido_logits, video_output_masks, video_output_boxes, video_output_embeds
         )):
            scores = logits.sigmoid().cpu().detach()  #[300,42]
            max_score, _ = torch.max(logits.sigmoid(),1)
            indices = torch.nonzero(max_score>self.inference_select_thres, as_tuple=False).squeeze(1)
            if len(indices) == 0:
                topkv, indices_top1 = torch.topk(scores.max(1)[0],k=1)
                indices_top1 = indices_top1[torch.argmax(topkv)]
                indices = [indices_top1.tolist()]
            else:
                nms_scores,idxs = torch.max(logits.sigmoid()[indices],1)
                boxes_before_nms = box_cxcywh_to_xyxy(output_boxes[indices])
                keep_indices = ops.batched_nms(boxes_before_nms,nms_scores,idxs,0.9)#.tolist()
                indices = indices[keep_indices]
            box_score = torch.max(logits.sigmoid()[indices],1)[0]
            det_bboxes = torch.cat([output_boxes[indices],box_score.unsqueeze(1)],dim=1)
            det_labels = torch.argmax(logits.sigmoid()[indices],dim=1)
            track_feats = output_embed[indices]
            ################################################
            if not isinstance(indices,list):
                indices=indices.cpu()
            ################################################
            det_masks = output_mask[indices]
            bboxes, labels, ids, indices = tracker.match(
            bboxes=det_bboxes,
            labels=det_labels,
            masks = det_masks,
            track_feats=track_feats,
            frame_id=i_frame,
            indices = indices)
            indices = torch.tensor(indices)[ids>-1].tolist()
            ids = ids[ids > -1]
            ids = ids.tolist()
            for query_i, id in zip(indices,ids):
                if id in video_dict.keys():
                    video_dict[id]['masks'].append(output_mask[query_i])
                    video_dict[id]['boxes'].append(output_boxes[query_i])
                    video_dict[id]['scores'].append(scores[query_i])
                    video_dict[id]['valid'] = video_dict[id]['valid'] + 1
                else:
                    video_dict[id] = {
                        'masks':[None for fi in range(i_frame)], 
                        'boxes':[None for fi in range(i_frame)], 
                        'scores':[None for fi in range(i_frame)], 
                        'valid':0}
                    video_dict[id]['masks'].append(output_mask[query_i])
                    video_dict[id]['boxes'].append(output_boxes[query_i])
                    video_dict[id]['scores'].append(scores[query_i])
                    video_dict[id]['valid'] = video_dict[id]['valid'] + 1

            for k,v in video_dict.items():
                if len(v['masks'])<i_frame+1: #padding None for unmatched ID
                    v['masks'].append(None)
                    v['scores'].append(None)
                    v['boxes'].append(None)
            check_len = [len(v['masks']) for k,v in video_dict.items()]
            # print('check_len',check_len)

            #  filtering sequences that are too short in video_dict (noise)，the rule is: if the first two frames are None and valid is less than 3
            if i_frame>8:
                del_list = []
                for k,v in video_dict.items():
                    if v['masks'][-1] is None and  v['masks'][-2] is None and v['valid']<3:
                        del_list.append(k)   
                for del_k in del_list:
                    video_dict.pop(del_k)                      

        del outputs
        logits_list = []
        masks_list = []

        for inst_id,m in  enumerate(video_dict.keys()):
            score_list_ori = video_dict[m]['scores']
            scores_temporal = []
            for k in score_list_ori:
                if k is not None:
                    scores_temporal.append(k)
            logits_i = torch.stack(scores_temporal)
            if self.temporal_score_type == 'mean':
                logits_i = logits_i.mean(0)
            elif self.temporal_score_type == 'max':
                logits_i = logits_i.max(0)[0]
            else:
                print('non valid temporal_score_type')
                import sys;sys.exit(0)
            logits_list.append(logits_i)
            
            # category_id = np.argmax(logits_i.mean(0))
            masks_list_i = []
            for n in range(vid_len):
                mask_i = video_dict[m]['masks'][n]
                if mask_i is None:    
                    zero_mask = None # padding None instead of zero mask to save memory
                    masks_list_i.append(zero_mask)
                else:
                    pred_mask_i =F.interpolate(mask_i[:,None,:,:],  size=(output_h*4, output_w*4) ,mode="bilinear", align_corners=False).sigmoid()
                    pred_mask_i = pred_mask_i[:,:,:image_sizes[0],:image_sizes[1]] #crop the padding area
                    pred_mask_i = (F.interpolate(pred_mask_i, size=(ori_size[0], ori_size[1]), mode='nearest')>0.5)[0,0].cpu() # resize to ori video size
                    masks_list_i.append(pred_mask_i)
            masks_list.append(masks_list_i)
        if len(logits_list)>0:
            pred_cls = torch.stack(logits_list)
        else:
            pred_cls = []

        if len(pred_cls) > 0:
            if self.is_multi_cls:
                is_above_thres = torch.where(pred_cls > self.apply_cls_thres)
                scores = pred_cls[is_above_thres]
                labels = is_above_thres[1]
                out_masks = [masks_list[valid_id] for valid_id in is_above_thres[0]]
            else:
                scores, labels = pred_cls.max(-1)
                out_masks = masks_list
            out_scores = scores.tolist()
            out_labels = labels.tolist()
        else:
            out_scores = []
            out_labels = []
            out_masks = []
        video_output = {
            "image_size": ori_size,
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
        }

        return video_output


    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(self.normalizer(frame.to(self.device)))
        images = ImageList.from_tensors(images)
        return images


    def coco_inference(self, box_cls, box_pred, mask_pred, image_sizes):
      
        assert len(box_cls) == len(image_sizes)
        results = []

        for i, (logits_per_image, box_pred_per_image, image_size) in enumerate(zip(
            box_cls, box_pred, image_sizes
        )):
            prob = logits_per_image.sigmoid()
            nms_scores,idxs = torch.max(prob,1)
            boxes_before_nms = box_cxcywh_to_xyxy(box_pred_per_image)
            keep_indices = ops.batched_nms(boxes_before_nms,nms_scores,idxs,0.7)  
            prob = prob[keep_indices]
            box_pred_per_image = box_pred_per_image[keep_indices]
            mask_pred_i = mask_pred[i][keep_indices]

            topk_values, topk_indexes = torch.topk(prob.view(-1), 100, dim=0)
            scores = topk_values
            topk_boxes = torch.div(topk_indexes, logits_per_image.shape[1], rounding_mode='floor')
            # topk_boxes = topk_indexes // logits_per_image.shape[1]
            labels = topk_indexes % logits_per_image.shape[1]
            scores_per_image = scores
            labels_per_image = labels

            box_pred_per_image = box_pred_per_image[topk_boxes]
            mask_pred_i = mask_pred_i[topk_boxes]

            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            if self.mask_on:
                N, C, H, W = mask_pred_i.shape
                mask = F.interpolate(mask_pred_i, size=(H*4, W*4), mode='bilinear', align_corners=False)
                mask = mask.sigmoid() > 0.5
                mask = mask[:,:,:image_size[0],:image_size[1]]
                result.pred_masks = mask

            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results



    def preprocess_coco_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images

    ###############################################################################
    #boxinst
    def add_bitmasks_from_boxes(self, instances, images, image_masks, im_h, im_w):
        # stride = self.mask_out_stride==4
        stride = 4
        start = int(stride // 2)

        assert images.size(2) % stride == 0
        assert images.size(3) % stride == 0

        downsampled_images = F.avg_pool2d(
            images.float(), kernel_size=stride,
            stride=stride, padding=0
        )[:, [2, 1, 0]]
        image_masks = image_masks[:, start::stride, start::stride]

        for im_i, per_im_gt_inst in enumerate(instances):
            images_lab = color.rgb2lab(downsampled_images[im_i].byte().permute(1, 2, 0).cpu().numpy())
            images_lab = torch.as_tensor(images_lab, device=downsampled_images.device, dtype=torch.float32)
            images_lab = images_lab.permute(2, 0, 1)[None]
            # self.pairwise_size=3
            # self.pairwise_dilation=2
            images_color_similarity = get_images_color_similarity(
                images_lab, image_masks[im_i],
                self.pairwise_size, self.pairwise_dilation
            )

            per_im_boxes = per_im_gt_inst.gt_boxes.tensor
            per_im_bitmasks = []
            per_im_bitmasks_full = []
            for per_box in per_im_boxes:
                bitmask_full = torch.zeros((im_h, im_w), device=self.device).float()
                bitmask_full[int(per_box[1]):int(per_box[3] + 1), int(per_box[0]):int(per_box[2] + 1)] = 1.0

                bitmask = bitmask_full[start::stride, start::stride]

                assert bitmask.size(0) * stride == im_h
                assert bitmask.size(1) * stride == im_w

                per_im_bitmasks.append(bitmask)
                per_im_bitmasks_full.append(bitmask_full)
            if len(per_im_boxes)==0:
                return
            per_im_gt_inst.gt_bitmasks = torch.stack(per_im_bitmasks, dim=0)
            per_im_gt_inst.gt_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0)
            per_im_gt_inst.image_color_similarity = torch.cat([
                images_color_similarity for _ in range(len(per_im_gt_inst))
            ], dim=0)
    ###############################################################################
    

##############################################################################
# boxinst
def get_images_color_similarity(images, image_masks, kernel_size, dilation):
    assert images.dim() == 4
    assert images.size(0) == 1

    unfolded_images = unfold_wo_center(
        images, kernel_size=kernel_size, dilation=dilation
    )

    diff = images[:, :, None] - unfolded_images
    similarity = torch.exp(-torch.norm(diff, dim=1) * 0.5)

    unfolded_weights = unfold_wo_center(
        image_masks[None, None], kernel_size=kernel_size,
        dilation=dilation
    )
    unfolded_weights = torch.max(unfolded_weights, dim=1)[0]

    return similarity * unfolded_weights


def unfold_wo_center(x, kernel_size, dilation):
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )

    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )

    # remove the center pixels
    size = kernel_size ** 2
    unfolded_x = torch.cat((
        unfolded_x[:, :, :size // 2],
        unfolded_x[:, :, size // 2 + 1:]
    ), dim=2)

    return unfolded_x

# def compute_locations(h, w, stride, device):
#     shifts_x = torch.arange(
#         0, w * stride, step=stride,
#         dtype=torch.float32, device=device
#     )
#     shifts_y = torch.arange(
#         0, h * stride, step=stride,
#         dtype=torch.float32, device=device
#     )
#     shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
#     shift_x = shift_x.reshape(-1)
#     shift_y = shift_y.reshape(-1)
#     locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
#     return locations
###############################################################################################




