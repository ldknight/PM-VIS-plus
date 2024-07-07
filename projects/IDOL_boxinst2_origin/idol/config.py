# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN


def add_idol_config(cfg):
    """
    Add config for IDOL.
    """
    cfg.MODEL.IDOL = CN()
    cfg.MODEL.IDOL.NUM_CLASSES = 80

    # DataLoader
    cfg.INPUT.SAMPLING_FRAME_NUM = 1
    cfg.INPUT.SAMPLING_FRAME_RANGE = 10
    cfg.INPUT.SAMPLING_INTERVAL = 1
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
    cfg.INPUT.AUGMENTATIONS = [] # "brightness", "contrast", "saturation", "rotation"

    cfg.INPUT.COCO_PRETRAIN = False
    cfg.INPUT.PRETRAIN_SAME_CROP = False
    ########################################################
    cfg.MODEL.PROPOSAL_GENERATOR.NAME = "FCOS"
    cfg.MODEL.FCOS = CN()
    cfg.MODEL.FCOS.NUM_CLASSES = 40
    cfg.MODEL.FCOS.IN_FEATURES = ["res5"]
    cfg.MODEL.FCOS.FPN_STRIDES = [ 32]
    cfg.MODEL.FCOS.PRIOR_PROB = 0.01
    cfg.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.05
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = 0.05
    cfg.MODEL.FCOS.NMS_TH = 0.6
    cfg.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 1000
    cfg.MODEL.FCOS.PRE_NMS_TOPK_TEST = 1000
    cfg.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 100
    cfg.MODEL.FCOS.POST_NMS_TOPK_TEST = 100
    cfg.MODEL.FCOS.TOP_LEVELS = 2
    cfg.MODEL.FCOS.NORM = "GN"  # Support GN or none
    cfg.MODEL.FCOS.USE_SCALE = True

    # The options for the quality of box prediction
    # It can be "ctrness" (as described in FCOS paper) or "iou"
    # Using "iou" here generally has ~0.4 better AP on COCO
    # Note that for compatibility, we still use the term "ctrness" in the code
    cfg.MODEL.FCOS.BOX_QUALITY = "ctrness"

    # Multiply centerness before threshold
    # This will affect the final performance by about 0.05 AP but save some time
    cfg.MODEL.FCOS.THRESH_WITH_CTR = True

    # Focal loss parameters
    cfg.MODEL.FCOS.LOSS_ALPHA = 0.25
    cfg.MODEL.FCOS.LOSS_GAMMA = 2.0

    # The normalizer of the classification loss
    # The normalizer can be "fg" (normalized by the number of the foreground samples),
    # "moving_fg" (normalized by the MOVING number of the foreground samples),
    # or "all" (normalized by the number of all samples)
    cfg.MODEL.FCOS.LOSS_NORMALIZER_CLS = "fg"
    cfg.MODEL.FCOS.LOSS_WEIGHT_CLS = 1.0

    cfg.MODEL.FCOS.SIZES_OF_INTEREST = [64, 128, 256, 512]
    cfg.MODEL.FCOS.USE_RELU = True
    cfg.MODEL.FCOS.USE_DEFORMABLE = False

    # the number of convolutions used in the cls and bbox tower
    cfg.MODEL.FCOS.NUM_CLS_CONVS = 4
    cfg.MODEL.FCOS.NUM_BOX_CONVS = 4
    cfg.MODEL.FCOS.NUM_SHARE_CONVS = 0
    cfg.MODEL.FCOS.CENTER_SAMPLE = True
    cfg.MODEL.FCOS.POS_RADIUS = 1.5
    cfg.MODEL.FCOS.LOC_LOSS_TYPE = 'giou'
    cfg.MODEL.FCOS.YIELD_PROPOSAL = False
    cfg.MODEL.FCOS.YIELD_BOX_FEATURES = False
    ########################################################
    # LOSS
    cfg.MODEL.IDOL.MASK_WEIGHT = 2.0
    cfg.MODEL.IDOL.DICE_WEIGHT = 5.0
    cfg.MODEL.IDOL.GIOU_WEIGHT = 2.0
    cfg.MODEL.IDOL.L1_WEIGHT = 5.0
    cfg.MODEL.IDOL.CLASS_WEIGHT = 2.0
    cfg.MODEL.IDOL.REID_WEIGHT = 2.0
    cfg.MODEL.IDOL.DEEP_SUPERVISION = True
    cfg.MODEL.IDOL.MASK_STRIDE = 4
    cfg.MODEL.IDOL.MATCH_STRIDE = 4
    cfg.MODEL.IDOL.FOCAL_ALPHA = 0.25

    cfg.MODEL.IDOL.SET_COST_CLASS = 2
    cfg.MODEL.IDOL.SET_COST_BOX = 5
    cfg.MODEL.IDOL.SET_COST_GIOU = 2

    # TRANSFORMER
    cfg.MODEL.IDOL.NHEADS = 8
    cfg.MODEL.IDOL.DROPOUT = 0.1
    cfg.MODEL.IDOL.DIM_FEEDFORWARD = 1024
    cfg.MODEL.IDOL.ENC_LAYERS = 6
    cfg.MODEL.IDOL.DEC_LAYERS = 6

    cfg.MODEL.IDOL.HIDDEN_DIM = 256
    cfg.MODEL.IDOL.NUM_OBJECT_QUERIES = 300
    cfg.MODEL.IDOL.DEC_N_POINTS = 4
    cfg.MODEL.IDOL.ENC_N_POINTS = 4
    cfg.MODEL.IDOL.NUM_FEATURE_LEVELS = 4


    # Evaluation
    cfg.MODEL.IDOL.CLIP_STRIDE = 1
    cfg.MODEL.IDOL.MERGE_ON_CPU = True
    cfg.MODEL.IDOL.MULTI_CLS_ON = True
    cfg.MODEL.IDOL.APPLY_CLS_THRES = 0.05

    cfg.MODEL.IDOL.TEMPORAL_SCORE_TYPE = 'mean' # mean or max score for sequence masks during inference,
    cfg.MODEL.IDOL.INFERENCE_SELECT_THRES = 0.1  # 0.05 for ytvis
    cfg.MODEL.IDOL.NMS_PRE =  0.5
    cfg.MODEL.IDOL.ADD_NEW_SCORE = 0.2
    cfg.MODEL.IDOL.INFERENCE_FW = True #frame weight
    cfg.MODEL.IDOL.INFERENCE_TW = True  #temporal weight
    cfg.MODEL.IDOL.MEMORY_LEN = 3
    cfg.MODEL.IDOL.BATCH_INFER_LEN = 10

    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    ## support Swin backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # find_unused_parameters
    cfg.FIND_UNUSED_PARAMETERS = True