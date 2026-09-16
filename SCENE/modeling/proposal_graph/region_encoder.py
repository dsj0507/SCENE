from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class ProposalRegionEncoder(nn.Module):
    def __init__(
        self,
        *,
        clip_embed_dim: int,
        remote_clip_embed_dim: int,
        fused_dim: int,
        clip_resolution: Tuple[int, int],
        remote_clip_resolution: Optional[Tuple[int, int]],
        clip_pixel_mean: torch.Tensor,
        clip_pixel_std: torch.Tensor,
        use_remote_clip: bool,
    ) -> None:
        super().__init__()
        self.clip_resolution = clip_resolution
        self.remote_clip_resolution = remote_clip_resolution or clip_resolution
        self.use_remote_clip = use_remote_clip

        self.register_buffer("clip_pixel_mean", clip_pixel_mean.clone(), persistent=False)
        self.register_buffer("clip_pixel_std", clip_pixel_std.clone(), persistent=False)

        self.geom_dim = 5
        self.clip_embed_dim = clip_embed_dim
        self.remote_clip_embed_dim = remote_clip_embed_dim if use_remote_clip else 0
        fusion_in_dim = self.clip_embed_dim + self.remote_clip_embed_dim + self.geom_dim
#几何特征嵌入
        self.geom_mlp = nn.Sequential(
            nn.Linear(self.geom_dim, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, fused_dim),
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in_dim, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, fused_dim),
        )

    def _resolve_model_resolution(
        self,
        model: nn.Module,
        fallback: Tuple[int, int],
    ) -> Tuple[int, int]:
        visual = getattr(model, "visual", None)
        input_resolution = getattr(visual, "input_resolution", None)
        if input_resolution is None:
            return fallback
        if isinstance(input_resolution, torch.Tensor):
            input_resolution = int(input_resolution.item())
        if isinstance(input_resolution, (tuple, list)):
            if len(input_resolution) == 2:
                return (int(input_resolution[0]), int(input_resolution[1]))
            if len(input_resolution) == 1:
                input_resolution = int(input_resolution[0])
        if isinstance(input_resolution, int):
            return (input_resolution, input_resolution)
        return fallback
#根据proposal的mask和box，从原图中裁剪出对应区域，调整到模型输入分辨率，并进行归一化处理，得到适合CLIP模型输入的特征图
    def _crop_region(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        box: torch.Tensor,
        resolution: Tuple[int, int],
    ) -> torch.Tensor:
        _, image_h, image_w = image.shape
        x0 = max(int(torch.floor(box[0]).item()), 0)
        y0 = max(int(torch.floor(box[1]).item()), 0)
        x1 = min(int(torch.ceil(box[2]).item()), image_w)
        y1 = min(int(torch.ceil(box[3]).item()), image_h)

        if x1 <= x0:
            x1 = min(x0 + 1, image_w)
        if y1 <= y0:
            y1 = min(y0 + 1, image_h)

        crop = image[:, y0:y1, x0:x1]
        crop_mask = mask[y0:y1, x0:x1].to(dtype=image.dtype).unsqueeze(0)
        masked_crop = crop * crop_mask
        resized = F.interpolate(
            masked_crop.unsqueeze(0),
            size=resolution,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        normalized = (resized - self.clip_pixel_mean) / self.clip_pixel_std
        return normalized
#计算5个几何特征：中心坐标（cx, cy），面积占比，宽高比，方向角（朝向），提供给后续的融合模块，帮助模型理解区域的空间属性和形状特征
    def _geometry_features(self, mask: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        image_h, image_w = mask.shape
        ys, xs = torch.nonzero(mask > 0, as_tuple=True)
        if ys.numel() == 0:
            return mask.new_zeros(self.geom_dim, dtype=torch.float32)

        xs = xs.to(torch.float32)
        ys = ys.to(torch.float32)
        cx = xs.mean() / max(float(image_w), 1.0)
        cy = ys.mean() / max(float(image_h), 1.0)
        area = float(xs.numel()) / max(float(image_h * image_w), 1.0)
        box_w = max(float(box[2] - box[0]), 1.0)
        box_h = max(float(box[3] - box[1]), 1.0)
        aspect_ratio = box_w / box_h

        x_centered = xs - xs.mean()
        y_centered = ys - ys.mean()
        cov_xx = (x_centered * x_centered).mean()
        cov_yy = (y_centered * y_centered).mean()
        cov_xy = (x_centered * y_centered).mean()
        orientation = 0.5 * torch.atan2(2.0 * cov_xy, cov_xx - cov_yy + 1e-6)
        orientation = orientation / torch.pi
#输出这个区域长啥样，在哪，多大，什么形状
        return torch.tensor(
            [cx, cy, area, aspect_ratio, orientation.item()],
            device=mask.device,
            dtype=torch.float32,
        )

    def forward(
        self,
        *,
        image: torch.Tensor,
        proposals: Dict[str, Any],
        clip_model: nn.Module,
        clip_model_remote: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        boxes = proposals["boxes"]
        masks = proposals["masks"]
        num_regions = boxes.shape[0]
        device = image.device
        clip_resolution = self._resolve_model_resolution(clip_model, self.clip_resolution)
        remote_clip_resolution = (
            self._resolve_model_resolution(clip_model_remote, self.remote_clip_resolution)
            if self.use_remote_clip and clip_model_remote is not None
            else self.remote_clip_resolution
        )

        if num_regions == 0:
            fused_dim = self.fusion_mlp[-1].out_features
            empty_remote = (
                torch.empty(0, self.remote_clip_embed_dim, device=device)
                if self.use_remote_clip
                else None
            )
            return {
                "clip_region_embeddings": torch.empty(0, self.clip_embed_dim, device=device),
                "remote_region_embeddings": empty_remote,
                "geometry_features": torch.empty(0, self.geom_dim, device=device),
                "geometry_embeddings": torch.empty(0, fused_dim, device=device),
                "fused_region_embeddings": torch.empty(0, fused_dim, device=device),
                "node_embeddings_raw": torch.empty(0, self.clip_embed_dim, device=device),
                "node_embeddings_fused": torch.empty(0, fused_dim, device=device),
                "node_geometry": torch.empty(0, self.geom_dim, device=device),
                "proposal_masks": proposals["masks"],
                "proposal_boxes": proposals["boxes"],
                "proposal_scores": proposals["scores"],
            }

        clip_crops = []
        remote_clip_crops = []
        geom_features = []
        for idx in range(num_regions):
            mask = masks[idx].squeeze(0) if masks[idx].ndim == 3 else masks[idx]
            box = boxes[idx]
            clip_crops.append(self._crop_region(image, mask, box, clip_resolution))
            if self.use_remote_clip and clip_model_remote is not None:
                remote_clip_crops.append(
                    self._crop_region(image, mask, box, remote_clip_resolution)
                )
            geom_features.append(self._geometry_features(mask, box))

        clip_crops = torch.stack(clip_crops, dim=0)
        geom_features = torch.stack(geom_features, dim=0)
##使用clip编码区域
        clip_region_embeddings = clip_model.encode_image(clip_crops)
        clip_region_embeddings = F.normalize(clip_region_embeddings.float(), dim=-1)

        remote_region_embeddings = None
        fusion_inputs = [clip_region_embeddings, geom_features]
        if self.use_remote_clip and clip_model_remote is not None:
            remote_clip_crops = torch.stack(remote_clip_crops, dim=0)
            remote_region_embeddings = clip_model_remote.encode_image(remote_clip_crops)
            remote_region_embeddings = F.normalize(remote_region_embeddings.float(), dim=-1)
            fusion_inputs.insert(1, remote_region_embeddings) #在fusion_inputs的位置1中插入remote_region_embeddings的向量值

        geometry_embeddings = self.geom_mlp(geom_features)
#融合后的特征包含 proposal 的视觉语义 + 遥感域视觉语义（可选）+ 空间几何先验
        fused_region_embeddings = self.fusion_mlp(torch.cat(fusion_inputs, dim=-1))

        return {
            "clip_region_embeddings": clip_region_embeddings,
            "remote_region_embeddings": remote_region_embeddings,
            "geometry_features": geom_features,
            "geometry_embeddings": geometry_embeddings,
            "fused_region_embeddings": fused_region_embeddings,
            "node_embeddings_raw": clip_region_embeddings,
            "node_embeddings_fused": fused_region_embeddings,
            "node_geometry": geom_features,
            "proposal_masks": proposals["masks"],
            "proposal_boxes": proposals["boxes"],
            "proposal_scores": proposals["scores"],
        }
