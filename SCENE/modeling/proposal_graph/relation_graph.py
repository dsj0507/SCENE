from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch.nn import functional as F


class SceneGraphBuilder:
    """Build a deterministic multi-relational graph from region proposals.

    Topology is classified once per unordered node pair with the priority
    containment > overlap > adjacent. Directional relations are then added
    independently for the x and y axes and may coexist with topology.
    """

    RELATION_NAMES = (
        "inside",
        "contains",
        "overlap",
        "adjacent",
        "left_of",
        "right_of",
        "above",
        "below",
    )
    NUM_RELATIONS = len(RELATION_NAMES)
    DEFAULT_THRESHOLDS = {
        "inside": 0.8,
        "overlap": 0.2,
        "adjacent": 0.01,
    }
    TOPOLOGY_RELATIONS = {"inside", "contains", "overlap", "adjacent"}
    DIRECTIONAL_RELATIONS = {"left_of", "right_of", "above", "below"}

    def __init__(self, *, thresholds: Dict[str, float] | None = None) -> None:
        self.relation_to_idx = {
            relation: idx for idx, relation in enumerate(self.RELATION_NAMES)
        }
        self.thresholds = dict(self.DEFAULT_THRESHOLDS)
        if thresholds is not None:
            self.thresholds.update(thresholds)

    def _pairwise_iou(self, masks: torch.Tensor) -> torch.Tensor:
        if masks.numel() == 0:
            return masks.new_zeros((0, 0), dtype=torch.float32)
        flat = masks.flatten(1).float()
        inter = flat @ flat.T
        area = flat.sum(dim=1, keepdim=True)
        union = area + area.T - inter
        return inter / union.clamp_min(1e-6)

    def _pairwise_inside(self, masks: torch.Tensor) -> torch.Tensor:
        if masks.numel() == 0:
            return masks.new_zeros((0, 0), dtype=torch.float32)
        flat = masks.flatten(1).float()
        inter = flat @ flat.T
        area_i = flat.sum(dim=1, keepdim=True)
        return inter / area_i.clamp_min(1e-6)

    def _centers_from_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes.new_empty((0, 2), dtype=torch.float32)
        return torch.stack(
            (
                (boxes[:, 0] + boxes[:, 2]) * 0.5,
                (boxes[:, 1] + boxes[:, 3]) * 0.5,
            ),
            dim=-1,
        )

    def _centers_from_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Return mask centroids in pixel coordinates as (cx, cy)."""
        num_nodes = masks.shape[0]
        centers = torch.zeros((num_nodes, 2), dtype=torch.float32, device=masks.device)
        for idx in range(num_nodes):
            ys, xs = torch.nonzero(masks[idx], as_tuple=True)
            if ys.numel() == 0:
                continue
            centers[idx, 0] = xs.float().mean()
            centers[idx, 1] = ys.float().mean()
        return centers

    def _mask_boundaries(self, masks: torch.Tensor) -> torch.Tensor:
        """Extract 8-neighbor mask boundaries using binary erosion."""
        if masks.numel() == 0:
            return masks.new_zeros(masks.shape, dtype=torch.bool)
        masks = masks.bool()
        height, width = masks.shape[-2:]
        padded = F.pad(masks.to(torch.uint8), (1, 1, 1, 1), value=0).bool()
        erosion = torch.ones_like(masks, dtype=torch.bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                erosion &= padded[
                    :, 1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width
                ]
        return masks & ~erosion

    def _pairwise_boundary_distance(
        self,
        masks: torch.Tensor,
        image_hw: Tuple[int, int],
        candidate_pairs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return normalized minimum Euclidean distance between mask boundaries.

        ``candidate_pairs`` is an optional symmetric boolean matrix.  When it
        is provided, distances are evaluated only for candidates that can
        still become adjacent.  This keeps the exact boundary-distance rule
        while avoiding a quadratic ``cdist`` over every proposal pair when
        containment or overlap has already won the topology priority.
        """
        num_nodes = masks.shape[0]
        if num_nodes == 0:
            return masks.new_empty((0, 0), dtype=torch.float32)

        boundaries = self._mask_boundaries(masks)
        boundary_points: List[torch.Tensor] = []
        for idx in range(num_nodes):
            ys, xs = torch.nonzero(boundaries[idx], as_tuple=True)
            if ys.numel() == 0:
                boundary_points.append(
                    torch.empty((0, 2), dtype=torch.float32, device=masks.device)
                )
            else:
                boundary_points.append(
                    torch.stack((xs.float(), ys.float()), dim=-1)
                )

        diagonal = max(float((image_hw[0] ** 2 + image_hw[1] ** 2) ** 0.5), 1.0)
        distances = torch.full(
            (num_nodes, num_nodes),
            float("inf"),
            dtype=torch.float32,
            device=masks.device,
        )
        distances.fill_diagonal_(0.0)
        for src in range(num_nodes):
            if boundary_points[src].numel() == 0:
                continue
            for dst in range(src + 1, num_nodes):
                if candidate_pairs is not None and not bool(candidate_pairs[src, dst]):
                    continue
                if boundary_points[dst].numel() == 0:
                    continue
                distance = (
                    torch.cdist(boundary_points[src], boundary_points[dst], p=2).min()
                    / diagonal
                )
                distances[src, dst] = distance
                distances[dst, src] = distance
        return distances

    def _pairwise_center_distance(
        self,
        centers: torch.Tensor,
        image_hw: Tuple[int, int],
    ) -> torch.Tensor:
        if centers.numel() == 0:
            return centers.new_zeros((0, 0), dtype=torch.float32)
        diagonal = max(float((image_hw[0] ** 2 + image_hw[1] ** 2) ** 0.5), 1.0)
        return torch.cdist(centers.float(), centers.float(), p=2) / diagonal

    def _pairwise_box_iou(self, boxes: torch.Tensor) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes.new_zeros((0, 0), dtype=torch.float32)
        lt = torch.maximum(boxes[:, None, :2], boxes[None, :, :2])
        rb = torch.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
        wh = (rb - lt).clamp_min(0.0)
        inter = wh[..., 0] * wh[..., 1]
        area = (
            (boxes[:, 2] - boxes[:, 0]).clamp_min(0.0)
            * (boxes[:, 3] - boxes[:, 1]).clamp_min(0.0)
        )
        union = area[:, None] + area[None, :] - inter
        return inter / union.clamp_min(1e-6)

    def _pairwise_box_gap(
        self,
        boxes: torch.Tensor,
        image_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """Retain box-gap diagnostics for backward compatibility only."""
        if boxes.numel() == 0:
            return boxes.new_zeros((0, 0), dtype=torch.float32)
        left_gap = boxes[:, None, 0] - boxes[None, :, 2]
        right_gap = boxes[None, :, 0] - boxes[:, None, 2]
        x_gap = torch.maximum(torch.maximum(left_gap, right_gap), boxes.new_zeros(()))
        top_gap = boxes[:, None, 1] - boxes[None, :, 3]
        bottom_gap = boxes[None, :, 1] - boxes[:, None, 3]
        y_gap = torch.maximum(torch.maximum(top_gap, bottom_gap), boxes.new_zeros(()))
        diagonal = max(float((image_hw[0] ** 2 + image_hw[1] ** 2) ** 0.5), 1.0)
        return torch.sqrt(x_gap.square() + y_gap.square()) / diagonal

    def _edge_continuous_geometry(
        self,
        boxes: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        pairwise_box_iou: torch.Tensor,
    ) -> torch.Tensor:
        eps = 1e-6
        if src.numel() == 0:
            return boxes.new_empty((0, 10), dtype=torch.float32)

        src_boxes = boxes[src].float()
        dst_boxes = boxes[dst].float()
        src_w = (src_boxes[:, 2] - src_boxes[:, 0]).clamp_min(eps)
        src_h = (src_boxes[:, 3] - src_boxes[:, 1]).clamp_min(eps)
        dst_w = (dst_boxes[:, 2] - dst_boxes[:, 0]).clamp_min(eps)
        dst_h = (dst_boxes[:, 3] - dst_boxes[:, 1]).clamp_min(eps)

        src_cx = (src_boxes[:, 0] + src_boxes[:, 2]) * 0.5
        src_cy = (src_boxes[:, 1] + src_boxes[:, 3]) * 0.5
        dst_cx = (dst_boxes[:, 0] + dst_boxes[:, 2]) * 0.5
        dst_cy = (dst_boxes[:, 1] + dst_boxes[:, 3]) * 0.5

        dx = (src_cx - dst_cx) / dst_w
        dy = (src_cy - dst_cy) / dst_h
        distance = torch.sqrt(dx.square() + dy.square())
        angle_sin = dy / distance.clamp_min(eps)
        angle_cos = dx / distance.clamp_min(eps)
        box_iou = pairwise_box_iou[src, dst]
        src_area = (src_w * src_h).clamp_min(eps)
        dst_area = (dst_w * dst_h).clamp_min(eps)
        area_ratio = torch.log((src_area / dst_area).clamp_min(eps))
        width_ratio = torch.log((src_w / dst_w).clamp_min(eps))
        height_ratio = torch.log((src_h / dst_h).clamp_min(eps))
        src_aspect = (src_w / src_h).clamp_min(eps)
        dst_aspect = (dst_w / dst_h).clamp_min(eps)
        aspect_ratio_diff = torch.log((src_aspect / dst_aspect).clamp_min(eps))

        return torch.stack(
            [
                dx,
                dy,
                distance,
                angle_sin,
                angle_cos,
                box_iou,
                area_ratio,
                width_ratio,
                height_ratio,
                aspect_ratio_diff,
            ],
            dim=-1,
        )

    def _add_edge(
        self,
        *,
        src: int,
        dst: int,
        relation: str,
        score: float | torch.Tensor,
        relation_to_idx: Dict[str, int],
        relation_mats: List[torch.Tensor],
        edge_sources: List[int],
        edge_targets: List[int],
        edge_relations: List[str],
        edge_scores: List[torch.Tensor],
        topology_edge_count: List[int],
        directional_edge_count: List[int],
    ) -> None:
        if relation not in relation_to_idx:
            return
        edge_sources.append(src)
        edge_targets.append(dst)
        edge_relations.append(relation)
        if torch.is_tensor(score):
            edge_scores.append(score.float())
        else:
            edge_scores.append(torch.tensor(float(score), dtype=torch.float32))
        relation_mats[relation_to_idx[relation]][src, dst] = 1.0
        if relation in self.TOPOLOGY_RELATIONS:
            topology_edge_count[0] += 1
        else:
            directional_edge_count[0] += 1

    @torch.cuda.amp.autocast(enabled=False)
    def build_single(
        self,
        *,
        proposals: Dict[str, Any],
        region_features: Dict[str, torch.Tensor] | None,
        image_hw: Tuple[int, int],
    ) -> Dict[str, Any]:
        # Large-mask intersections exceed FP16's finite range under outer AMP.
        # Graph geometry is inexpensive and must stay in FP32 to avoid inf / inf.
        boxes = proposals["boxes"].float()
        masks = proposals["masks"]
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        masks = masks.to(dtype=torch.bool)
        scores = proposals["scores"].float()
        num_nodes = boxes.shape[0]
        device = boxes.device if num_nodes > 0 else scores.device

        if region_features is not None:
            fused_region_embeddings = region_features["fused_region_embeddings"]
            geometry_features = region_features["geometry_features"]
        else:
            fused_region_embeddings = torch.empty((num_nodes, 0), device=device)
            geometry_features = torch.zeros((num_nodes, 5), device=device)

        node_areas = masks.flatten(1).sum(dim=1).float()
        node_centers = self._centers_from_masks(masks)
        node_centers_norm = node_centers.clone()
        node_centers_norm[:, 0] /= max(float(image_hw[1]), 1.0)
        node_centers_norm[:, 1] /= max(float(image_hw[0]), 1.0)
        node_box_centers = self._centers_from_boxes(boxes)

        pairwise_iou = self._pairwise_iou(masks)
        pairwise_inside = self._pairwise_inside(masks)
        pairwise_box_iou = self._pairwise_box_iou(boxes)
        pairwise_box_gap = self._pairwise_box_gap(boxes, image_hw)
        pairwise_distance = self._pairwise_center_distance(node_centers, image_hw)

        relation_to_idx = self.relation_to_idx
        relation_mats = [
            torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
            for _ in range(self.NUM_RELATIONS)
        ]
        edge_sources: List[int] = []
        edge_targets: List[int] = []
        edge_relations: List[str] = []
        edge_scores: List[torch.Tensor] = []
        topology_pair_count = 0
        topology_edge_count = [0]
        directional_edge_count = [0]

        inside_threshold = float(self.thresholds["inside"])
        overlap_threshold = float(self.thresholds["overlap"])
        adjacent_threshold = float(self.thresholds["adjacent"])
        image_height, image_width = image_hw

        # Containment and overlap have higher topology priority than
        # adjacency.  Bounding-box gap is a lower bound on mask-boundary
        # distance, so only unresolved pairs that can reach the adjacency
        # threshold need an exact boundary-distance calculation.
        containment_pairs = (
            (pairwise_inside >= inside_threshold)
            | (pairwise_inside.T >= inside_threshold)
        )
        boundary_candidates = (
            (pairwise_iou < overlap_threshold)
            & ~containment_pairs
            & (pairwise_box_gap <= adjacent_threshold)
        )
        boundary_candidates.fill_diagonal_(False)
        pairwise_boundary_distance = self._pairwise_boundary_distance(
            masks,
            image_hw,
            candidate_pairs=boundary_candidates,
        )

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                ratio_i = pairwise_inside[i, j]
                ratio_j = pairwise_inside[j, i]
                iou = pairwise_iou[i, j]
                boundary_distance = pairwise_boundary_distance[i, j]

                # Topology is classified once for the unordered pair {i, j}.
                if ratio_i >= inside_threshold and ratio_j < inside_threshold:
                    topology_pair_count += 1
                    self._add_edge(
                        src=i,
                        dst=j,
                        relation="inside",
                        score=ratio_i,
                        relation_to_idx=relation_to_idx,
                        relation_mats=relation_mats,
                        edge_sources=edge_sources,
                        edge_targets=edge_targets,
                        edge_relations=edge_relations,
                        edge_scores=edge_scores,
                        topology_edge_count=topology_edge_count,
                        directional_edge_count=directional_edge_count,
                    )
                    self._add_edge(
                        src=j,
                        dst=i,
                        relation="contains",
                        score=ratio_i,
                        relation_to_idx=relation_to_idx,
                        relation_mats=relation_mats,
                        edge_sources=edge_sources,
                        edge_targets=edge_targets,
                        edge_relations=edge_relations,
                        edge_scores=edge_scores,
                        topology_edge_count=topology_edge_count,
                        directional_edge_count=directional_edge_count,
                    )
                elif ratio_j >= inside_threshold and ratio_i < inside_threshold:
                    topology_pair_count += 1
                    self._add_edge(
                        src=j,
                        dst=i,
                        relation="inside",
                        score=ratio_j,
                        relation_to_idx=relation_to_idx,
                        relation_mats=relation_mats,
                        edge_sources=edge_sources,
                        edge_targets=edge_targets,
                        edge_relations=edge_relations,
                        edge_scores=edge_scores,
                        topology_edge_count=topology_edge_count,
                        directional_edge_count=directional_edge_count,
                    )
                    self._add_edge(
                        src=i,
                        dst=j,
                        relation="contains",
                        score=ratio_j,
                        relation_to_idx=relation_to_idx,
                        relation_mats=relation_mats,
                        edge_sources=edge_sources,
                        edge_targets=edge_targets,
                        edge_relations=edge_relations,
                        edge_scores=edge_scores,
                        topology_edge_count=topology_edge_count,
                        directional_edge_count=directional_edge_count,
                    )
                elif ratio_i >= inside_threshold and ratio_j >= inside_threshold:
                    topology_pair_count += 1
                    self._add_edge(
                        src=i,
                        dst=j,
                        relation="overlap",
                        score=iou,
                        relation_to_idx=relation_to_idx,
                        relation_mats=relation_mats,
                        edge_sources=edge_sources,
                        edge_targets=edge_targets,
                        edge_relations=edge_relations,
                        edge_scores=edge_scores,
                        topology_edge_count=topology_edge_count,
                        directional_edge_count=directional_edge_count,
                    )
                    self._add_edge(
                        src=j,
                        dst=i,
                        relation="overlap",
                        score=iou,
                        relation_to_idx=relation_to_idx,
                        relation_mats=relation_mats,
                        edge_sources=edge_sources,
                        edge_targets=edge_targets,
                        edge_relations=edge_relations,
                        edge_scores=edge_scores,
                        topology_edge_count=topology_edge_count,
                        directional_edge_count=directional_edge_count,
                    )
                elif iou >= overlap_threshold:
                    topology_pair_count += 1
                    self._add_edge(
                        src=i,
                        dst=j,
                        relation="overlap",
                        score=iou,
                        relation_to_idx=relation_to_idx,
                        relation_mats=relation_mats,
                        edge_sources=edge_sources,
                        edge_targets=edge_targets,
                        edge_relations=edge_relations,
                        edge_scores=edge_scores,
                        topology_edge_count=topology_edge_count,
                        directional_edge_count=directional_edge_count,
                    )
                    self._add_edge(
                        src=j,
                        dst=i,
                        relation="overlap",
                        score=iou,
                        relation_to_idx=relation_to_idx,
                        relation_mats=relation_mats,
                        edge_sources=edge_sources,
                        edge_targets=edge_targets,
                        edge_relations=edge_relations,
                        edge_scores=edge_scores,
                        topology_edge_count=topology_edge_count,
                        directional_edge_count=directional_edge_count,
                    )
                elif boundary_distance <= adjacent_threshold:
                    topology_pair_count += 1
                    adjacent_score = (
                        torch.exp(-boundary_distance / adjacent_threshold)
                        if adjacent_threshold > 0
                        else torch.ones((), device=device)
                    )
                    self._add_edge(
                        src=i,
                        dst=j,
                        relation="adjacent",
                        score=adjacent_score,
                        relation_to_idx=relation_to_idx,
                        relation_mats=relation_mats,
                        edge_sources=edge_sources,
                        edge_targets=edge_targets,
                        edge_relations=edge_relations,
                        edge_scores=edge_scores,
                        topology_edge_count=topology_edge_count,
                        directional_edge_count=directional_edge_count,
                    )
                    self._add_edge(
                        src=j,
                        dst=i,
                        relation="adjacent",
                        score=adjacent_score,
                        relation_to_idx=relation_to_idx,
                        relation_mats=relation_mats,
                        edge_sources=edge_sources,
                        edge_targets=edge_targets,
                        edge_relations=edge_relations,
                        edge_scores=edge_scores,
                        topology_edge_count=topology_edge_count,
                        directional_edge_count=directional_edge_count,
                    )

                # Direction is independent on the x and y axes.
                dx = node_centers[i, 0] - node_centers[j, 0]
                dy = node_centers[i, 1] - node_centers[j, 1]
                dx_score = dx.abs() / max(float(image_width), 1.0)
                dy_score = dy.abs() / max(float(image_height), 1.0)

                if dx < 0:
                    horizontal_pair = (("left_of", i, j), ("right_of", j, i))
                elif dx > 0:
                    horizontal_pair = (("right_of", i, j), ("left_of", j, i))
                else:
                    horizontal_pair = ()
                for relation, src, dst in horizontal_pair:
                    self._add_edge(
                        src=src,
                        dst=dst,
                        relation=relation,
                        score=dx_score,
                        relation_to_idx=relation_to_idx,
                        relation_mats=relation_mats,
                        edge_sources=edge_sources,
                        edge_targets=edge_targets,
                        edge_relations=edge_relations,
                        edge_scores=edge_scores,
                        topology_edge_count=topology_edge_count,
                        directional_edge_count=directional_edge_count,
                    )

                if dy < 0:
                    vertical_pair = (("above", i, j), ("below", j, i))
                elif dy > 0:
                    vertical_pair = (("below", i, j), ("above", j, i))
                else:
                    vertical_pair = ()
                for relation, src, dst in vertical_pair:
                    self._add_edge(
                        src=src,
                        dst=dst,
                        relation=relation,
                        score=dy_score,
                        relation_to_idx=relation_to_idx,
                        relation_mats=relation_mats,
                        edge_sources=edge_sources,
                        edge_targets=edge_targets,
                        edge_relations=edge_relations,
                        edge_scores=edge_scores,
                        topology_edge_count=topology_edge_count,
                        directional_edge_count=directional_edge_count,
                    )

        if edge_sources:
            src = torch.tensor(edge_sources, dtype=torch.long, device=device)
            dst = torch.tensor(edge_targets, dtype=torch.long, device=device)
            edge_index = torch.stack((src, dst), dim=0)
            edge_type = torch.tensor(
                [relation_to_idx[name] for name in edge_relations],
                dtype=torch.long,
                device=device,
            )
            edge_score = torch.stack(edge_scores).to(device=device)
            edge_geom_delta = geometry_features[src] - geometry_features[dst]
            edge_geometry = self._edge_continuous_geometry(
                boxes,
                src,
                dst,
                pairwise_box_iou,
            )
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_type = torch.empty((0,), dtype=torch.long, device=device)
            edge_score = torch.empty((0,), dtype=torch.float32, device=device)
            edge_geom_delta = torch.empty((0, 5), dtype=torch.float32, device=device)
            edge_geometry = torch.empty((0, 10), dtype=torch.float32, device=device)

        relation_tensor = (
            torch.stack(relation_mats, dim=0)
            if relation_mats
            else torch.empty((0, num_nodes, num_nodes), dtype=torch.float32, device=device)
        )
        num_total_edges = len(edge_sources)
        topology_edges = topology_edge_count[0]
        directional_edges = directional_edge_count[0]
        relation_edge_counts = torch.zeros(
            self.NUM_RELATIONS,
            dtype=torch.long,
            device=device,
        )
        if edge_relations:
            relation_edge_counts = torch.bincount(
                torch.tensor(
                    [relation_to_idx[name] for name in edge_relations],
                    dtype=torch.long,
                    device=device,
                ),
                minlength=self.NUM_RELATIONS,
            )
        relation_edge_ratios = relation_edge_counts.float() / max(num_total_edges, 1)

        return {
            "relation_names": self.RELATION_NAMES,
            "num_nodes": torch.tensor([num_nodes], dtype=torch.long, device=device),
            "node_masks": masks,
            "node_boxes": boxes,
            "node_areas": node_areas,
            "node_scores": scores,
            "node_centers": node_centers,
            "node_centers_norm": node_centers_norm,
            "node_box_centers": node_box_centers,
            "node_embeddings": fused_region_embeddings,
            "node_geometry": geometry_features,
            "edge_index": edge_index,
            "edge_type": edge_type,
            "edge_score": edge_score,
            "edge_geom_delta": edge_geom_delta,
            "edge_geometry": edge_geometry,
            "relation_tensor": relation_tensor,
            "pairwise_iou": pairwise_iou,
            "pairwise_inside": pairwise_inside,
            "pairwise_boundary_distance": pairwise_boundary_distance,
            "pairwise_box_iou": pairwise_box_iou,
            "pairwise_box_gap": pairwise_box_gap,
            "pairwise_distance": pairwise_distance,
            "num_topology_pairs": torch.tensor(
                topology_pair_count, dtype=torch.long, device=device
            ),
            "num_topology_edges": torch.tensor(
                topology_edges, dtype=torch.long, device=device
            ),
            "num_directional_edges": torch.tensor(
                directional_edges, dtype=torch.long, device=device
            ),
            "directional_edges_before_pruning": torch.tensor(
                directional_edges, dtype=torch.long, device=device
            ),
            "directional_edges_after_pruning": torch.tensor(
                directional_edges, dtype=torch.long, device=device
            ),
            "num_total_edges": torch.tensor(
                num_total_edges, dtype=torch.long, device=device
            ),
            "relation_edge_counts": relation_edge_counts,
            "relation_edge_ratios": relation_edge_ratios,
            "average_out_degree": torch.tensor(
                num_total_edges / max(num_nodes, 1),
                dtype=torch.float32,
                device=device,
            ),
            "average_in_degree": torch.tensor(
                num_total_edges / max(num_nodes, 1),
                dtype=torch.float32,
                device=device,
            ),
        }

    def build_batch(
        self,
        graph_items: Iterable[Dict[str, Any]],
    ) -> Dict[str, Any]:
        items = list(graph_items)
        if not items:
            return {
                "relation_names": self.RELATION_NAMES,
                "node_embeddings": torch.empty(0, 0),
                "node_geometry": torch.empty(0, 5),
                "node_boxes": torch.empty(0, 4),
                "node_areas": torch.empty(0),
                "node_scores": torch.empty(0),
                "node_centers": torch.empty(0, 2),
                "node_centers_norm": torch.empty(0, 2),
                "node_box_centers": torch.empty(0, 2),
                "edge_index": torch.empty(2, 0, dtype=torch.long),
                "edge_type": torch.empty(0, dtype=torch.long),
                "edge_score": torch.empty(0),
                "edge_geom_delta": torch.empty(0, 5),
                "edge_geometry": torch.empty(0, 10),
                "node_batch_index": torch.empty(0, dtype=torch.long),
                "graph_ptr": torch.tensor([0], dtype=torch.long),
                "graph_num_nodes": torch.empty(0, dtype=torch.long),
                "graph_num_topology_pairs": torch.empty(0, dtype=torch.long),
                "graph_num_topology_edges": torch.empty(0, dtype=torch.long),
                "graph_num_directional_edges": torch.empty(0, dtype=torch.long),
                "graph_num_directional_edges_before_pruning": torch.empty(
                    0, dtype=torch.long
                ),
                "graph_num_directional_edges_after_pruning": torch.empty(
                    0, dtype=torch.long
                ),
                "graph_num_total_edges": torch.empty(0, dtype=torch.long),
                "graph_relation_edge_counts": torch.empty(
                    (0, self.NUM_RELATIONS), dtype=torch.long
                ),
                "graph_relation_edge_ratios": torch.empty(
                    (0, self.NUM_RELATIONS), dtype=torch.float32
                ),
            }

        device = items[0]["node_boxes"].device
        node_boxes = []
        node_areas = []
        node_scores = []
        node_centers = []
        node_centers_norm = []
        node_box_centers = []
        node_geometry = []
        node_embeddings = []
        node_batch_index = []
        edge_index = []
        edge_type = []
        edge_score = []
        edge_geom_delta = []
        edge_geometry = []
        graph_num_nodes = []
        graph_num_topology_pairs = []
        graph_num_topology_edges = []
        graph_num_directional_edges = []
        graph_num_directional_edges_before_pruning = []
        graph_num_directional_edges_after_pruning = []
        graph_num_total_edges = []
        graph_relation_edge_counts = []
        graph_relation_edge_ratios = []
        graph_ptr = [0]
        offset = 0

        for graph_idx, item in enumerate(items):
            num_nodes = int(item["num_nodes"].item())
            node_boxes.append(item["node_boxes"])
            node_areas.append(item["node_areas"])
            node_scores.append(item["node_scores"])
            node_centers.append(item["node_centers"])
            node_centers_norm.append(item["node_centers_norm"])
            node_box_centers.append(item["node_box_centers"])
            node_geometry.append(item["node_geometry"])
            node_embeddings.append(item["node_embeddings"])
            node_batch_index.append(
                torch.full((num_nodes,), graph_idx, dtype=torch.long, device=device)
            )
            graph_num_nodes.append(num_nodes)
            graph_num_topology_pairs.append(int(item["num_topology_pairs"].item()))
            graph_num_topology_edges.append(int(item["num_topology_edges"].item()))
            graph_num_directional_edges.append(int(item["num_directional_edges"].item()))
            graph_num_directional_edges_before_pruning.append(
                int(item["directional_edges_before_pruning"].item())
            )
            graph_num_directional_edges_after_pruning.append(
                int(item["directional_edges_after_pruning"].item())
            )
            graph_num_total_edges.append(int(item["num_total_edges"].item()))
            graph_relation_edge_counts.append(item["relation_edge_counts"])
            graph_relation_edge_ratios.append(item["relation_edge_ratios"])
            if item["edge_index"].numel() > 0:
                edge_index.append(item["edge_index"] + offset)
                edge_type.append(item["edge_type"])
                edge_score.append(item["edge_score"])
                edge_geom_delta.append(item["edge_geom_delta"])
                edge_geometry.append(item["edge_geometry"])
            offset += num_nodes
            graph_ptr.append(offset)

        return {
            "relation_names": self.RELATION_NAMES,
            "node_embeddings": torch.cat(node_embeddings, dim=0),
            "node_geometry": torch.cat(node_geometry, dim=0),
            "node_boxes": torch.cat(node_boxes, dim=0),
            "node_areas": torch.cat(node_areas, dim=0),
            "node_scores": torch.cat(node_scores, dim=0),
            "node_centers": torch.cat(node_centers, dim=0),
            "node_centers_norm": torch.cat(node_centers_norm, dim=0),
            "node_box_centers": torch.cat(node_box_centers, dim=0),
            "edge_index": (
                torch.cat(edge_index, dim=1)
                if edge_index
                else torch.empty((2, 0), dtype=torch.long, device=device)
            ),
            "edge_type": (
                torch.cat(edge_type, dim=0)
                if edge_type
                else torch.empty((0,), dtype=torch.long, device=device)
            ),
            "edge_score": (
                torch.cat(edge_score, dim=0)
                if edge_score
                else torch.empty((0,), dtype=torch.float32, device=device)
            ),
            "edge_geom_delta": (
                torch.cat(edge_geom_delta, dim=0)
                if edge_geom_delta
                else torch.empty((0, 5), dtype=torch.float32, device=device)
            ),
            "edge_geometry": (
                torch.cat(edge_geometry, dim=0)
                if edge_geometry
                else torch.empty((0, 10), dtype=torch.float32, device=device)
            ),
            "node_batch_index": torch.cat(node_batch_index, dim=0),
            "graph_ptr": torch.tensor(graph_ptr, dtype=torch.long, device=device),
            "graph_num_nodes": torch.tensor(graph_num_nodes, dtype=torch.long, device=device),
            "graph_num_topology_pairs": torch.tensor(
                graph_num_topology_pairs, dtype=torch.long, device=device
            ),
            "graph_num_topology_edges": torch.tensor(
                graph_num_topology_edges, dtype=torch.long, device=device
            ),
            "graph_num_directional_edges": torch.tensor(
                graph_num_directional_edges, dtype=torch.long, device=device
            ),
            "graph_num_directional_edges_before_pruning": torch.tensor(
                graph_num_directional_edges_before_pruning,
                dtype=torch.long,
                device=device,
            ),
            "graph_num_directional_edges_after_pruning": torch.tensor(
                graph_num_directional_edges_after_pruning,
                dtype=torch.long,
                device=device,
            ),
            "graph_num_total_edges": torch.tensor(
                graph_num_total_edges, dtype=torch.long, device=device
            ),
            "graph_relation_edge_counts": torch.stack(graph_relation_edge_counts),
            "graph_relation_edge_ratios": torch.stack(graph_relation_edge_ratios),
        }
