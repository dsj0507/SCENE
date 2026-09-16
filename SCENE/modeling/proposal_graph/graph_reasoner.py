from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .relation_graph import SceneGraphBuilder


class RelationEncoder(nn.Module):
    def __init__(
        self,
        *,
        geometry_dim: int = 10,
        edge_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_relations = SceneGraphBuilder.NUM_RELATIONS
        self.geometry_dim = int(geometry_dim)
        self.edge_dim = int(edge_dim)
        self.relation_embedding = nn.Embedding(self.num_relations, self.edge_dim)
        self.geometry_mlp = nn.Sequential(
            nn.Linear(self.geometry_dim, self.edge_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.edge_dim, self.edge_dim),
            nn.ReLU(inplace=True),
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(self.edge_dim * 2, self.edge_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.edge_dim, self.edge_dim),
        )

    def forward(self, edge_type: torch.Tensor, edge_geometry: torch.Tensor) -> torch.Tensor:
        if edge_type.numel() == 0:
            return edge_geometry.new_empty((0, self.edge_dim))
        relation_feat = self.relation_embedding(edge_type.long())
        geometry_feat = self.geometry_mlp(edge_geometry.float())
        return self.output_mlp(torch.cat([relation_feat, geometry_feat], dim=-1))


class RelationCompatibilityScorer(nn.Module):
    def __init__(
        self,
        *,
        text_dim: int,
        edge_dim: int,
        hidden_dim: int,
        edge_chunk_size: int = 2,#原来是8
        target_chunk_size: int = 8,#原来是32
    ) -> None:
        super().__init__()
        self.text_dim = int(text_dim)
        self.edge_dim = int(edge_dim)
        self.edge_chunk_size = max(int(edge_chunk_size), 1)
        self.target_chunk_size = max(int(target_chunk_size), 1)
        self.mlp = nn.Sequential(
            nn.Linear(self.text_dim * 3 + self.edge_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def _score_chunk(
        self,
        *,
        src_text: torch.Tensor,
        target_text: torch.Tensor,
        edge_feat: torch.Tensor,
    ) -> torch.Tensor:
        edge_count = edge_feat.shape[0]
        src_count = src_text.shape[0]
        target_count = target_text.shape[0]
        src = src_text.view(1, src_count, 1, self.text_dim).expand(edge_count, src_count, target_count, -1)
        target = target_text.view(1, 1, target_count, self.text_dim).expand(edge_count, src_count, target_count, -1)
        product = src * target
        edge = edge_feat.view(edge_count, 1, 1, self.edge_dim).expand(edge_count, src_count, target_count, -1)
        scorer_input = torch.cat([src, target, product, edge], dim=-1)
        return self.mlp(scorer_input).squeeze(-1)

    def forward(
        self,
        *,
        node_logits: torch.Tensor,
        text_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            node_logits.numel() == 0
            or text_embeddings.numel() == 0
            or edge_index.numel() == 0
            or edge_feat.numel() == 0
        ):
            return node_logits.new_zeros(node_logits.shape), edge_feat

        text_embeddings = F.normalize(text_embeddings.float(), dim=-1)
        src_idx, dst_idx = edge_index[0], edge_index[1]
        src_probs = F.softmax(node_logits.detach(), dim=-1)[src_idx]
        num_nodes, num_classes = node_logits.shape
        context_bias = node_logits.new_zeros((num_nodes, num_classes))

        for edge_start in range(0, edge_feat.shape[0], self.edge_chunk_size):
            edge_end = min(edge_start + self.edge_chunk_size, edge_feat.shape[0])
            edge_slice = edge_feat[edge_start:edge_end]
            prob_slice = src_probs[edge_start:edge_end]
            dst_slice = dst_idx[edge_start:edge_end]

            for target_start in range(0, num_classes, self.target_chunk_size):
                target_end = min(target_start + self.target_chunk_size, num_classes)
                scores = self._score_chunk(
                    src_text=text_embeddings,
                    target_text=text_embeddings[target_start:target_end],
                    edge_feat=edge_slice,
                )
                context_part = (prob_slice.unsqueeze(-1) * scores).sum(dim=1)
                context_part = F.pad(context_part, (target_start, num_classes - target_end))
                context_part = context_part.to(dtype=context_bias.dtype)
                context_bias.index_add_(
                    0,
                    dst_slice,
                    context_part,
                )

        return context_bias, edge_feat


class SceneGraphReasoner(nn.Module):
    RELATION_GROUP_NAMES = (
        "containment",
        "horizontal",
        "vertical",
        "overlap",
        "adjacent",
    )
    RELATION_GROUP_IDS = (
        0,  # inside
        0,  # contains
        3,  # overlap
        4,  # adjacent
        1,  # left_of
        1,  # right_of
        2,  # above
        2,  # below
    )

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_relations = SceneGraphBuilder.NUM_RELATIONS
        self.num_relation_groups = len(self.RELATION_GROUP_NAMES)
        self.num_layers = max(int(num_layers), 1)
        self.node_geometry_dim = 5
        self.edge_geometry_dim = 10

        self.node_projection = nn.Linear(hidden_dim, hidden_dim)
        self.geometry_projection = nn.Linear(self.node_geometry_dim, hidden_dim)
        self.relation_embedding = nn.Embedding(self.num_relations, hidden_dim)

        self.group_source_projections = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(hidden_dim, hidden_dim, bias=False)
                        for _ in range(self.num_relation_groups)
                    ]
                )
                for _ in range(self.num_layers)
            ]
        )
        self.group_target_projections = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(hidden_dim, hidden_dim, bias=False)
                        for _ in range(self.num_relation_groups)
                    ]
                )
                for _ in range(self.num_layers)
            ]
        )
        self.confidence_projection = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attention_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        hidden_dim * 3 + self.edge_geometry_dim + 3,
                        hidden_dim,
                    ),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(self.num_layers)
            ]
        )
        self.update_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(self.num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.num_layers)])

    def _parameter_count(self) -> Tuple[int, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        return total, trainable

    def _parameter_diagnostics(self) -> Dict[str, int]:
        total, trainable = self._parameter_count()
        hidden_dim = int(self.hidden_dim)
        relation_count = int(self.num_relations)
        group_count = int(self.num_relation_groups)
        layer_count = int(self.num_layers)
        current_group_projection_count = (
            layer_count * 2 * group_count * hidden_dim * hidden_dim
        )
        confidence_parameter_count = sum(
            parameter.numel() for parameter in self.confidence_projection.parameters()
        )
        legacy_relation_projection_count = (
            layer_count * 2 * relation_count * hidden_dim * hidden_dim
        )
        legacy_classifier_count = (
            hidden_dim * (hidden_dim * 3 + self.edge_geometry_dim)
            + hidden_dim
            + relation_count * hidden_dim
            + relation_count
        )
        estimated_pre_refactor_count = (
            total
            - confidence_parameter_count
            + legacy_relation_projection_count
            - current_group_projection_count
            + legacy_classifier_count
        )
        return {
            "parameter_count": total,
            "trainable_parameter_count": trainable,
            "relation_group_projection_parameter_count": current_group_projection_count,
            "confidence_parameter_count": confidence_parameter_count,
            "legacy_relation_projection_parameter_count": legacy_relation_projection_count,
            "legacy_relation_classifier_parameter_count": legacy_classifier_count,
            "estimated_pre_refactor_parameter_count": estimated_pre_refactor_count,
            "parameter_count_delta": total - estimated_pre_refactor_count,
        }

    def _forward_shapes(
        self,
        graph_batch: Dict[str, torch.Tensor],
        *,
        initial_node_embeddings: torch.Tensor,
        refined_node_embeddings: torch.Tensor,
    ) -> Dict[str, Tuple[int, ...]]:
        shapes: Dict[str, Tuple[int, ...]] = {}
        for key in (
            "node_embeddings",
            "node_geometry",
            "node_scores",
            "edge_index",
            "edge_type",
            "edge_score",
            "edge_geometry",
        ):
            value = graph_batch.get(key)
            if torch.is_tensor(value):
                shapes[key] = tuple(value.shape)
        shapes["initial_node_embeddings"] = tuple(initial_node_embeddings.shape)
        shapes["node_embeddings_refined"] = tuple(refined_node_embeddings.shape)
        return shapes

    def _get_node_confidence(
        self,
        graph_batch: Dict[str, torch.Tensor],
        *,
        num_nodes: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node_scores = graph_batch.get("node_scores")
        if node_scores is None or node_scores.numel() == 0:
            node_scores = torch.ones(num_nodes, dtype=dtype, device=device)
        else:
            node_scores = node_scores.to(device=device, dtype=dtype).reshape(-1)
            if node_scores.shape[0] != num_nodes:
                raise ValueError(
                    "node_scores must contain one value per node: "
                    f"got {node_scores.shape[0]} for {num_nodes} nodes"
                )
            node_scores = node_scores.clamp(0.0, 1.0)
        confidence_embedding = self.confidence_projection(
            node_scores.unsqueeze(-1)
        )
        return node_scores, confidence_embedding

    def _copy_graph_diagnostics(
        self,
        graph_batch: Dict[str, torch.Tensor],
        outputs: Dict[str, Any],
    ) -> None:
        for key in (
            "graph_num_nodes",
            "graph_num_topology_pairs",
            "graph_num_topology_edges",
            "graph_num_directional_edges",
            "graph_num_directional_edges_before_pruning",
            "graph_num_directional_edges_after_pruning",
            "graph_num_total_edges",
            "graph_relation_edge_counts",
            "graph_relation_edge_ratios",
        ):
            if key in graph_batch:
                outputs[key] = graph_batch[key]

    def _empty_reasoning(self, graph_batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        edge_index = graph_batch["edge_index"]
        edge_type = graph_batch["edge_type"]
        device = graph_batch["node_boxes"].device
        empty_nodes = torch.empty((0, self.hidden_dim), device=device)
        parameter_diagnostics = self._parameter_diagnostics()
        outputs: Dict[str, Any] = {
            "relation_names": SceneGraphBuilder.RELATION_NAMES,
            "node_embeddings": empty_nodes,
            "node_embeddings_refined": empty_nodes,
            "refined_region_embeddings": empty_nodes,
            "initial_node_embeddings": empty_nodes,
            "edge_index": edge_index,
            "edge_type": edge_type,
            "edge_score": graph_batch["edge_score"],
            "edge_geom_delta": graph_batch.get("edge_geom_delta"),
            "edge_geometry": graph_batch.get("edge_geometry"),
            "node_batch_index": graph_batch["node_batch_index"],
            "graph_ptr": graph_batch["graph_ptr"],
            "node_confidence": torch.empty((0,), device=device),
            "forward_shapes": self._forward_shapes(
                graph_batch,
                initial_node_embeddings=empty_nodes,
                refined_node_embeddings=empty_nodes,
            ),
        }
        outputs.update(parameter_diagnostics)
        self._copy_graph_diagnostics(graph_batch, outputs)
        return outputs

    def _get_edge_features(self, graph_batch: Dict[str, torch.Tensor], edge_mask: torch.Tensor) -> torch.Tensor:
        edge_geometry = graph_batch.get("edge_geometry")
        if edge_geometry is not None and edge_geometry.numel() > 0:
            return edge_geometry[edge_mask].float()
        edge_geom_delta = graph_batch.get("edge_geom_delta")
        if edge_geom_delta is not None and edge_geom_delta.numel() > 0:
            return edge_geom_delta[edge_mask].float()
        return graph_batch["node_boxes"].new_empty((int(edge_mask.sum().item()), 0))

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, graph_batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        # Edge messages are accumulated before degree normalization. Keep this
        # small graph path in FP32 so dense graphs cannot overflow FP16 first.
        node_embeddings = graph_batch["node_embeddings"]
        if node_embeddings.numel() == 0:
            return self._empty_reasoning(graph_batch)

        node_geometry = graph_batch["node_geometry"]
        edge_index = graph_batch["edge_index"]
        edge_type = graph_batch["edge_type"]
        edge_score = graph_batch["edge_score"]

        node_scores, confidence_embedding = self._get_node_confidence(
            graph_batch,
            num_nodes=node_embeddings.shape[0],
            dtype=torch.float32,
            device=node_embeddings.device,
        )
        x = (
            self.node_projection(node_embeddings.float())
            + self.geometry_projection(node_geometry.float())
            + confidence_embedding
        )
        x0 = x

        for layer_idx, (update_mlp, norm) in enumerate(zip(self.update_mlps, self.norms)):
            aggregated = x.new_zeros(x.shape)
            gate_mass = x.new_zeros((x.shape[0], 1))
            if edge_index.numel() > 0:
                for rel_idx in range(self.num_relations):
                    rel_mask = edge_type == rel_idx
                    if not torch.any(rel_mask):
                        continue
                    rel_edge_index = edge_index[:, rel_mask]
                    src_idx, dst_idx = rel_edge_index[0], rel_edge_index[1]
                    group_idx = self.RELATION_GROUP_IDS[rel_idx]
                    rel_src = self.group_source_projections[layer_idx][group_idx](
                        x[src_idx]
                    )
                    rel_dst = self.group_target_projections[layer_idx][group_idx](
                        x[dst_idx]
                    )
                    rel_embed = self.relation_embedding(
                        torch.full_like(src_idx, rel_idx, dtype=torch.long)
                    ).to(dtype=x.dtype)
                    rel_edge_score = edge_score[rel_mask].float().unsqueeze(-1).to(
                        dtype=x.dtype
                    )
                    edge_features = self._get_edge_features(
                        graph_batch, rel_mask
                    ).to(dtype=x.dtype)
                    source_confidence = node_scores[src_idx].unsqueeze(-1).to(
                        dtype=x.dtype
                    )
                    target_confidence = node_scores[dst_idx].unsqueeze(-1).to(
                        dtype=x.dtype
                    )
                    attention_input = torch.cat(
                        [
                            x[src_idx],
                            x[dst_idx],
                            rel_embed,
                            edge_features,
                            rel_edge_score,
                            source_confidence,
                            target_confidence,
                        ],
                        dim=-1,
                    )
                    attention = torch.sigmoid(
                        self.attention_mlps[layer_idx](attention_input)
                    )
                    relation_message = (
                        (rel_src + rel_dst + rel_embed) * attention
                    ).to(dtype=aggregated.dtype)
                    aggregated.index_add_(0, dst_idx, relation_message)
                    gate_mass.index_add_(
                        0,
                        dst_idx,
                        attention.to(dtype=gate_mass.dtype),
                    )

                # Preserve all relation messages while preventing high-degree
                # nodes from receiving a proportionally larger update magnitude.
                # The lower bound avoids amplifying weak total gate mass.
                aggregated = aggregated / gate_mass.clamp_min(1.0)

            update = update_mlp(torch.cat([x, aggregated], dim=-1))
            x = norm(x + update)

        parameter_diagnostics = self._parameter_diagnostics()
        outputs: Dict[str, Any] = {
            "relation_names": SceneGraphBuilder.RELATION_NAMES,
            "node_embeddings": x,
            "node_embeddings_refined": x,
            "refined_region_embeddings": x,
            "initial_node_embeddings": x0,
            "edge_index": edge_index,
            "edge_type": edge_type,
            "edge_score": graph_batch["edge_score"],
            "edge_geom_delta": graph_batch.get("edge_geom_delta"),
            "edge_geometry": graph_batch.get("edge_geometry"),
            "node_batch_index": graph_batch["node_batch_index"],
            "graph_ptr": graph_batch["graph_ptr"],
            "node_confidence": node_scores,
            "forward_shapes": self._forward_shapes(
                graph_batch,
                initial_node_embeddings=x0,
                refined_node_embeddings=x,
            ),
        }
        outputs.update(parameter_diagnostics)
        self._copy_graph_diagnostics(graph_batch, outputs)
        return outputs

    def to_dense_guidance(
        self,
        *,
        proposals: Sequence[Dict[str, Any]],
        reasoning_outputs: Dict[str, torch.Tensor],
        target_hw: Tuple[int, int],
        node_features: torch.Tensor | None = None,
        node_weights: torch.Tensor | Sequence[torch.Tensor] | None = None,
    ) -> Dict[str, Any]:
        base_node_features = reasoning_outputs.get("node_embeddings_refined")
        if node_features is None:
            node_features = base_node_features if base_node_features is not None else reasoning_outputs["node_embeddings"]
        device = node_features.device
        hidden_dim = node_features.shape[-1] if node_features.ndim == 2 else self.hidden_dim
        graph_ptr = reasoning_outputs["graph_ptr"]
        h, w = target_hw

        dense_maps: List[torch.Tensor] = []
        confidence_maps: List[torch.Tensor] = []
        normalizer_maps: List[torch.Tensor] = []

        for graph_idx, proposal_item in enumerate(proposals):
            start = int(graph_ptr[graph_idx].item())
            end = int(graph_ptr[graph_idx + 1].item())
            if end <= start:
                dense_maps.append(torch.zeros(hidden_dim, h, w, device=device))
                confidence_maps.append(torch.zeros(1, h, w, device=device))
                normalizer_maps.append(torch.zeros(1, h, w, device=device))
                continue

            masks = proposal_item["masks"]
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            masks = F.interpolate(
                masks.float().unsqueeze(1),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

            if node_weights is None:
                weights = proposal_item["scores"].float()
            elif torch.is_tensor(node_weights):
                weights = node_weights[start:end].float()
            else:
                weights = node_weights[graph_idx].float()
            weighted_masks = masks * weights.view(-1, 1, 1)

            node_slice = node_features[start:end]
            dense_map = torch.einsum("nhw,nc->chw", weighted_masks, node_slice)
            normalizer = weighted_masks.sum(dim=0, keepdim=True).clamp_min(1e-6)
            dense_map = dense_map / normalizer

            confidence_map = weighted_masks.max(dim=0, keepdim=True).values
            dense_maps.append(dense_map)
            confidence_maps.append(confidence_map)
            normalizer_maps.append(normalizer)

        dense_guidance = (
            torch.stack(dense_maps, dim=0)
            if dense_maps
            else torch.empty((0, hidden_dim, h, w), device=device)
        )
        confidence_guidance = (
            torch.stack(confidence_maps, dim=0)
            if confidence_maps
            else torch.empty((0, 1, h, w), device=device)
        )
        normalizer_guidance = (
            torch.stack(normalizer_maps, dim=0)
            if normalizer_maps
            else torch.empty((0, 1, h, w), device=device)
        )

        return {
            "dense_guidance": dense_guidance,
            "confidence_guidance": confidence_guidance,
            "normalizer_guidance": normalizer_guidance,
            "target_hw": torch.tensor(target_hw, dtype=torch.long, device=device),
        }

    def build_guidance_pyramid(
        self,
        dense_guidance: torch.Tensor,
        *,
        feature_sizes: Iterable[Tuple[int, int]],
    ) -> List[torch.Tensor]:
        return [
            F.interpolate(dense_guidance, size=feature_hw, mode="bilinear", align_corners=False)
            for feature_hw in feature_sizes
        ]
