# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union
from einops import rearrange

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY


from ..transformer.scene_predictor import SCENEPredictor


@SEM_SEG_HEADS_REGISTRY.register()
class SCENEHead(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        num_classes: int,
        ignore_value: int = -1,
        # extra parameters
        feature_resolution: list,
        transformer_predictor: nn.Module,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            num_classes: number of classes to predict
            ignore_value: category id to be ignored during training.
            feature_resolution: resolution of the feature map
            transformer_predictor: the transformer decoder that makes prediction
        """
        super().__init__()
        self.ignore_value = ignore_value
        self.predictor = transformer_predictor
        self.num_classes = num_classes
        self.feature_resolution = feature_resolution

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "feature_resolution": cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION,
            "transformer_predictor": SCENEPredictor(
                cfg,
            ),
        }

    def forward(
        self,
        features,
        dino_feat,
        guidance_features,
        guidance_features_remote,
        dino_guidance_feat,
        prompt=None,
        gt_cls=None,
        graph_state=None,
    ):
        """
        Arguments:
            img_feats: (B, C, HW)
            guidance_features: (B, C, )
        """
        if isinstance(features, list):
            img_feat0 = rearrange(features[0][:, 1:, :], "b (h w) c->b c h w", h=self.feature_resolution[0], w=self.feature_resolution[1])
            img_feat1 = rearrange(features[1][:, 1:, :], "b (h w) c->b c h w", h=self.feature_resolution[0], w=self.feature_resolution[1])
            img_feat2 = rearrange(features[2][:, 1:, :], "b (h w) c->b c h w", h=self.feature_resolution[0], w=self.feature_resolution[1])
            img_feat3 = rearrange(features[3][:, 1:, :], "b (h w) c->b c h w", h=self.feature_resolution[0], w=self.feature_resolution[1])
            img_feat = [img_feat0,img_feat1,img_feat2,img_feat3]
        elif torch.is_tensor(features) and features.dim() == 4:
            img_feat = features
        else:
            img_feat = rearrange(features[:, 1:, :], "b (h w) c->b c h w", h=self.feature_resolution[0], w=self.feature_resolution[1])

        outputs = self.predictor(
            img_feat,
            dino_feat,
            guidance_features,
            guidance_features_remote,
            dino_guidance_feat,
            prompt,
            gt_cls,
            graph_state=graph_state,
        )
        if isinstance(outputs, dict):
            return outputs
        return {"logits": outputs}
    
