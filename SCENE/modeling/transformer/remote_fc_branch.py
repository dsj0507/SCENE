from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from .EfficientAggregator.model_agg_spatial import SptialFusionBlock
from .OriAggregator.model_agg_spatial import SwinTransformerBlockWrapper


class ContentAdaptiveReceptiveFieldRouter(nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        fine_kernels: Sequence[int],
        coarse_kernels: Sequence[int],
    ) -> None:
        super().__init__()
        self.fine_kernels = tuple(int(k) for k in fine_kernels)
        self.coarse_kernels = tuple(int(k) for k in coarse_kernels)
        if len(self.fine_kernels) != 3 or len(self.coarse_kernels) != 3:
            raise ValueError(
                "ContentAdaptiveReceptiveFieldRouter requires exactly three "
                "fine kernels and three coarse kernels."
            )
        self.fine_router = nn.Conv2d(in_channels, 3, kernel_size=1, stride=1, padding=0)
        self.coarse_router = nn.Conv2d(in_channels, 3, kernel_size=1, stride=1, padding=0)

    def _pool_bank(self, x: torch.Tensor, kernels: Sequence[int]) -> torch.Tensor:
        pooled = []
        for kernel_size in kernels:
            pooled.append(
                F.avg_pool2d(
                    x,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    count_include_pad=False,
                )
            )
        return torch.stack(pooled, dim=1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fine_bank = self._pool_bank(x, self.fine_kernels)
        coarse_bank = self._pool_bank(x, self.coarse_kernels)

        fine_weights = torch.softmax(self.fine_router(x), dim=1)
        coarse_weights = torch.softmax(self.coarse_router(x), dim=1)
        fine_feat = (fine_bank * fine_weights.unsqueeze(2)).sum(dim=1)
        coarse_feat = (coarse_bank * coarse_weights.unsqueeze(2)).sum(dim=1)
        return fine_feat, coarse_feat, fine_weights, coarse_weights


class EfficientSpatialStack(nn.Module):
    def __init__(self, hidden_dim: int, *, num_heads: int, depth: int, sr_ratio: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [SptialFusionBlock(hidden_dim, num_heads, sr_ratio=sr_ratio) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, guidance)
        return x


class OriSpatialStack(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        num_heads: int,
        depth: int,
        input_resolution: Tuple[int, int],
        window_size: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SwinTransformerBlockWrapper(
                    hidden_dim,
                    hidden_dim,
                    input_resolution,
                    nheads=num_heads,
                    window_size=window_size,
                    pad_len=0,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(2)
        for layer in self.layers:
            x = layer(x, guidance)
        return x.squeeze(2)


class RemoteFineCoarseBranch(nn.Module):
    def __init__(
        self,
        *,
        remote_guidance_dims: Sequence[int],
        branch_dim: int,
        num_heads: int,
        use_efficient: bool,
        feature_resolution: Tuple[int, int],
        decoder_dims: Sequence[int],
        window_size: int,
        graph_dense_dim: int,
        fine_kernels: Sequence[int],
        coarse_kernels: Sequence[int],
        fine_depth: int,
        coarse_depth: int,
    ) -> None:
        super().__init__()
        self.branch_dim = branch_dim

        self.remote_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, branch_dim, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                )
                for in_channels in remote_guidance_dims
            ]
        )
        self.base_fusion = nn.Sequential(
            nn.Conv2d(branch_dim * len(remote_guidance_dims), branch_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_dim, branch_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.receptive_field_router = ContentAdaptiveReceptiveFieldRouter(
            branch_dim,
            fine_kernels=fine_kernels,
            coarse_kernels=coarse_kernels,
        )
        decoder_resolution = tuple(
            int(dim * (2 ** len(decoder_dims))) for dim in feature_resolution
        )
        fine_window = max(int(window_size) // 2, 4)
        coarse_window = max(int(window_size), fine_window)

        if use_efficient:
            self.fine_module = EfficientSpatialStack(
                branch_dim,
                num_heads=num_heads,
                depth=fine_depth,
                sr_ratio=1,
            )
            self.coarse_module = EfficientSpatialStack(
                branch_dim,
                num_heads=num_heads,
                depth=coarse_depth,
                sr_ratio=4,
            )
        else:
            self.fine_module = OriSpatialStack(
                branch_dim,
                num_heads=num_heads,
                depth=fine_depth,
                input_resolution=decoder_resolution,
                window_size=fine_window,
            )
            self.coarse_module = OriSpatialStack(
                branch_dim,
                num_heads=num_heads,
                depth=coarse_depth,
                input_resolution=decoder_resolution,
                window_size=coarse_window,
            )

        self.fc_fusion = nn.Sequential(
            nn.Conv2d(branch_dim * 3, branch_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_dim, branch_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.graph_feature_projection = nn.Sequential(
            nn.Conv2d(graph_dense_dim, branch_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.remote_from_graph = nn.Sequential(
            nn.Conv2d(branch_dim * 2, branch_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_dim, branch_dim, kernel_size=3, stride=1, padding=1),
        )

    def _build_graph_feature(
        self,
        graph_dense_feature: torch.Tensor | None,
        graph_context_coverage: torch.Tensor | None,
        *,
        target_hw: Tuple[int, int],
    ) -> torch.Tensor | None:
        if graph_dense_feature is None:
            return None

        graph_feature = self.graph_feature_projection(graph_dense_feature)
        graph_feature = F.interpolate(
            graph_feature,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )
        if graph_context_coverage is not None:
            graph_gate = F.interpolate(
                graph_context_coverage,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            ).clamp_(0.0, 1.0)
            graph_feature = graph_feature * graph_gate

        return graph_feature

    def forward(
        self,
        *,
        appearance_guidance_remote,
        graph_dense_feature: torch.Tensor | None,
        graph_context_coverage: torch.Tensor | None,
        target_hw: Tuple[int, int],
    ) -> Dict[str, torch.Tensor | None]:
        if appearance_guidance_remote is None:
            return {
                "remote_base_feature": None,
                "remote_fc_feature": None,
                "remote_graph_enhanced": None,
                "fine_routing_weights": None,
                "coarse_routing_weights": None,
            }

        projected_features = []
        for projection, feature in zip(self.remote_projections, appearance_guidance_remote):
            if feature is None:
                continue
            projected = projection(feature)
            projected = F.interpolate(
                projected,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
            projected_features.append(projected)

        if not projected_features:
            return {
                "remote_base_feature": None,
                "remote_fc_feature": None,
                "remote_graph_enhanced": None,
                "fine_routing_weights": None,
                "coarse_routing_weights": None,
            }

        remote_base_feature = (
            projected_features[0]
            if len(projected_features) == 1
            else self.base_fusion(torch.cat(projected_features, dim=1))
        )

        (
            remote_fine_feat,
            remote_coarse_feat,
            fine_routing_weights,
            coarse_routing_weights,
        ) = self.receptive_field_router(remote_base_feature)
        remote_fine_feat = self.fine_module(remote_fine_feat, remote_base_feature)
        remote_coarse_feat = self.coarse_module(remote_coarse_feat, remote_base_feature)
        remote_fc_feature = remote_base_feature + self.fc_fusion(
            torch.cat([remote_base_feature, remote_fine_feat, remote_coarse_feat], dim=1)
        )

        remote_graph_enhanced = remote_fc_feature
        graph_feature = self._build_graph_feature(
            graph_dense_feature,
            graph_context_coverage,
            target_hw=target_hw,
        )

        if graph_feature is not None:
            cross_input = torch.cat([remote_fc_feature, graph_feature], dim=1)
            remote_updated = remote_fc_feature + self.remote_from_graph(cross_input)
            remote_graph_enhanced = remote_updated

        return {
            "remote_base_feature": remote_base_feature,
            "remote_fc_feature": remote_fc_feature,
            "remote_graph_enhanced": remote_graph_enhanced,
            "fine_routing_weights": fine_routing_weights,
            "coarse_routing_weights": coarse_routing_weights,
        }
