# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_scene_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"

    cfg.DATASETS.VAL_ALL = ("coco_2017_val_all_stuff_sem_seg",)

    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # swin transformer backbone
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
    
    # zero shot config
    cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON = "datasets/ADE20K_2021_17_01/ADE20K_847.json"
    cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON = "datasets/ADE20K_2021_17_01/ADE20K_847.json"
    cfg.MODEL.SEM_SEG_HEAD.USE_CLIP_CORR = True
    cfg.MODEL.SEM_SEG_HEAD.DECODER_DIMS = [128,64] 
    cfg.MODEL.SEM_SEG_HEAD.DECODER_CLIP_GUIDANCE_DIMS = [0,0] 
    cfg.MODEL.SEM_SEG_HEAD.DECODER_CLIP_GUIDANCE_PROJ_DIMS = [0,0] 
    cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED = "ViT-B/16"
    cfg.MODEL.SEM_SEG_HEAD.IMAGE_ENCODER = "clip"
    # Controls which CLIP space encodes SAM3 proposal crops.
    # auto uses both CLIP encoders when RemoteCLIP is enabled.
    cfg.MODEL.SEM_SEG_HEAD.REGION_ENCODER_SOURCE = "auto"
    # rotate
    cfg.MODEL.SEM_SEG_HEAD.USE_ROTATE = True
    # clip_remote
    cfg.MODEL.SEM_SEG_HEAD.USE_REMOTE_CLIP = True
    cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED_WEIGHTS = "pretrained/ViT-B-16.pt"
    cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED_REMOTE = "ViT-B/32"
    cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED_WEIGHTS_REMOTE = "pretrained/RemoteCLIP-ViT-B-32.pt"
    # DINO is used only as the main image encoder when IMAGE_ENCODER == "dino".
    cfg.MODEL.SEM_SEG_HEAD.DINO_FINETUNE = "freeze"
    cfg.MODEL.SEM_SEG_HEAD.DINO_WEIGHTS = ""
    cfg.MODEL.SEM_SEG_HEAD.USE_SAM3_PROPOSAL = True
    cfg.MODEL.SEM_SEG_HEAD.SAM3_CHECKPOINT_PATH = "pretrained/sam3.pt"
    cfg.MODEL.SEM_SEG_HEAD.SAM3_BPE_PATH = "pretrained/bpe_simple_vocab_16e6.txt.gz"
    cfg.MODEL.SEM_SEG_HEAD.SAM3_CONFIDENCE_THRESHOLD = 0.5
    cfg.MODEL.SEM_SEG_HEAD.SAM3_PROMPT_MODE = "visual_text"
    cfg.MODEL.SEM_SEG_HEAD.SAM3_GEOMETRY_GRID_SIZES = [1, 2, 4]
    cfg.MODEL.SEM_SEG_HEAD.SAM3_GEOMETRY_INCLUDE_STRIPES = True
    cfg.MODEL.SEM_SEG_HEAD.SAM3_GEOMETRY_STRIPE_COUNT = 4
    cfg.MODEL.SEM_SEG_HEAD.SAM3_GEOMETRY_INCLUDE_CENTER = True
    cfg.MODEL.SEM_SEG_HEAD.SAM3_GEOMETRY_NMS_IOU_THRESHOLD = 0.85
    cfg.MODEL.SEM_SEG_HEAD.SAM3_GEOMETRY_BOX_SCALE = 1.15
    cfg.MODEL.SEM_SEG_HEAD.SAM3_GEOMETRY_MAX_PROMPTS = 32
    cfg.MODEL.SEM_SEG_HEAD.ENABLE_PROPOSAL_ANALYSIS = False
    cfg.MODEL.SEM_SEG_HEAD.PROPOSAL_MAX_NUM = 64
    cfg.MODEL.SEM_SEG_HEAD.PROPOSAL_MIN_AREA = 32
    cfg.MODEL.SEM_SEG_HEAD.PROPOSAL_MATCH_IOU_THRESHOLD = 0.2
    cfg.MODEL.SEM_SEG_HEAD.PROPOSAL_MATCH_COVERAGE_THRESHOLD = 0.5
    cfg.MODEL.SEM_SEG_HEAD.PROPOSAL_QUALITY_SUPERVISION = False
    cfg.MODEL.SEM_SEG_HEAD.PROPOSAL_QUALITY_LOSS_WEIGHTING = True
    cfg.MODEL.SEM_SEG_HEAD.PROPOSAL_HARD_QUALITY_THRESHOLD = 0.16
    cfg.MODEL.SEM_SEG_HEAD.PROPOSAL_WEAK_QUALITY_THRESHOLD = 0.08
    cfg.MODEL.SEM_SEG_HEAD.PROPOSAL_HARD_MIN_IOU = 0.0
    cfg.MODEL.SEM_SEG_HEAD.PROPOSAL_WEAK_GRAPH_SUPERVISION = False
    cfg.MODEL.SEM_SEG_HEAD.PROPOSAL_QUALITY_FORMULA = "sqrt_iou_coverage"
    cfg.MODEL.SEM_SEG_HEAD.VL_LOSS_WEIGHT = 0.25
    cfg.MODEL.SEM_SEG_HEAD.VL_LOSS_TEMPERATURE = 0.07
    cfg.MODEL.SEM_SEG_HEAD.VL_LOSS_USE_FUSED_REGION = True
    cfg.MODEL.SEM_SEG_HEAD.GRAPH_INSIDE_THRESHOLD = 0.8
    cfg.MODEL.SEM_SEG_HEAD.GRAPH_OVERLAP_THRESHOLD = 0.2
    cfg.MODEL.SEM_SEG_HEAD.GRAPH_ADJACENT_THRESHOLD = 0.01
    cfg.MODEL.SEM_SEG_HEAD.GRAPH_HIDDEN_DIM = 256
    cfg.MODEL.SEM_SEG_HEAD.GRAPH_NUM_LAYERS = 2
    cfg.MODEL.SEM_SEG_HEAD.GRAPH_MODE = "graph_context_fusion"
    cfg.MODEL.SEM_SEG_HEAD.USE_NODE_TEXT_MATCH = True
    cfg.MODEL.SEM_SEG_HEAD.LAMBDA_CONTEXT = 0.05
    cfg.MODEL.SEM_SEG_HEAD.CONTEXT_BIAS_SCALE = 1.0
    # Backward-compatible flag kept so older yaml files still merge cleanly.
    # The current decoder always uses graph-context fusion when GRAPH_MODE enables it.
    cfg.MODEL.SEM_SEG_HEAD.NODE_MATCH_WEIGHT = 1.0
    cfg.MODEL.SEM_SEG_HEAD.NODE_PSEUDO_WEIGHT = 1.0
    cfg.MODEL.SEM_SEG_HEAD.NODE_ORIGINAL_AUX_WEIGHT = 0.5
    cfg.MODEL.SEM_SEG_HEAD.NODE_HARD_LOSS_CAP = 3.0
    cfg.MODEL.SEM_SEG_HEAD.NODE_CONSISTENCY_WEIGHT = 0.05
    cfg.MODEL.SEM_SEG_HEAD.NODE_GRAPH_CONSISTENCY_WEIGHT = 0.25
    cfg.MODEL.SEM_SEG_HEAD.ENABLE_NODE_PSEUDO_LOSS = True
    cfg.MODEL.SEM_SEG_HEAD.NODE_PSEUDO_START_EPOCH = 500
    # Fixed residual weight for raw CLIP image-text cosine logits.
    cfg.MODEL.SEM_SEG_HEAD.RAW_CLIP_LOGIT_WEIGHT = 1.0
    cfg.MODEL.SEM_SEG_HEAD.LOGIT_OUTPUT_MODE = "fused"
    cfg.MODEL.SEM_SEG_HEAD.USE_SOFT_REGION_CONTEXT = True
    # Keep gated dense graph-context injection and allow independent ablation
    # of graph context inside remote_fc_branch.
    cfg.MODEL.SEM_SEG_HEAD.USE_GRAPH_CONTEXT_IN_REMOTE_FC = True
    cfg.MODEL.SEM_SEG_HEAD.NODE_PSEUDO_TOPK = 3
    cfg.MODEL.SEM_SEG_HEAD.NODE_PSEUDO_TEMPERATURE = 0.07
    cfg.MODEL.SEM_SEG_HEAD.NODE_PSEUDO_CONFIDENCE = 0.85
    cfg.MODEL.SEM_SEG_HEAD.NODE_PSEUDO_MARGIN = 0.2
    cfg.MODEL.SEM_SEG_HEAD.USE_REMOTE_FC_DECODER = True
    cfg.MODEL.SEM_SEG_HEAD.REMOTE_FC_BRANCH_DIM = 32
    cfg.MODEL.SEM_SEG_HEAD.REMOTE_FC_FINE_KERNELS = [3, 5, 7]
    cfg.MODEL.SEM_SEG_HEAD.REMOTE_FC_COARSE_KERNELS = [7, 9, 11]
    cfg.MODEL.SEM_SEG_HEAD.REMOTE_FC_FINE_DEPTH = 1
    cfg.MODEL.SEM_SEG_HEAD.REMOTE_FC_COARSE_DEPTH = 1

    cfg.MODEL.PROMPT_ENSEMBLE = False
    cfg.MODEL.PROMPT_ENSEMBLE_TYPE = "single"

    cfg.MODEL.CLIP_PIXEL_MEAN = [122.7709383, 116.7460125, 104.09373615]
    cfg.MODEL.CLIP_PIXEL_STD = [68.5005327, 66.6321579, 70.3231630]
    # three styles for clip classification, crop, mask, cropmask

    cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM = 512
    cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_PROJ_DIM = 128
    cfg.MODEL.SEM_SEG_HEAD.DECODER_DIMS = [64, 32]
    cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_DIMS = [256, 128]

    cfg.MODEL.SEM_SEG_HEAD.USE_EFFICIENT = True
    cfg.MODEL.SEM_SEG_HEAD.NUM_HEADS = 4
    cfg.MODEL.SEM_SEG_HEAD.HIDDEN_DIMS = 128
    cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION = [24, 24]
    cfg.MODEL.SEM_SEG_HEAD.WINDOW_SIZES = 12

    cfg.MODEL.SEM_SEG_HEAD.PROMPT_DEPTH = 0
    cfg.MODEL.SEM_SEG_HEAD.PROMPT_LENGTH = 0
    cfg.SOLVER.CLIP_MULTIPLIER = 0.01
    cfg.LOG_PERIOD = 20

    cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE = "attention"
    cfg.TEST.SLIDING_WINDOW = False
