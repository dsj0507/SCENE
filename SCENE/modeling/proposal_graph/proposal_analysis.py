"""Lightweight, CPU-side statistics for proposal-pipeline analysis.

The helpers in this module deliberately return plain Python values. They are
diagnostic-only and must not keep proposal tensors alive or participate in
model computation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import torch


HISTOGRAM_EDGES = tuple(i / 10.0 for i in range(11))


def _as_cpu_float(values: torch.Tensor) -> torch.Tensor:
    return values.detach().float().reshape(-1).cpu()


def _quantile(values: torch.Tensor, fraction: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values, fraction).item())


def summarize_values(values: torch.Tensor, *, histogram_edges=HISTOGRAM_EDGES) -> Dict[str, Any]:
    values = _as_cpu_float(values)
    if values.numel() == 0:
        summary = {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
        }
    else:
        summary = {
            "count": int(values.numel()),
            "min": float(values.min().item()),
            "max": float(values.max().item()),
            "mean": float(values.mean().item()),
            "std": float(values.std(unbiased=False).item()),
            "p10": _quantile(values, 0.10),
            "p25": _quantile(values, 0.25),
            "p50": _quantile(values, 0.50),
            "p75": _quantile(values, 0.75),
            "p90": _quantile(values, 0.90),
        }

    edges = [float(edge) for edge in histogram_edges]
    if len(edges) < 2:
        raise ValueError("histogram_edges must contain at least two values")
    if values.numel() == 0:
        counts = [0] * (len(edges) - 1)
    else:
        clipped = values.clamp(edges[0], edges[-1])
        counts = torch.histc(
            clipped,
            bins=len(edges) - 1,
            min=edges[0],
            max=edges[-1],
        ).long().tolist()
        # torch.histc includes the right-most endpoint in the final bin. The
        # fixed bins are diagnostic only, but this correction keeps counts
        # exact for values equal to the lower edge of a clipped final bin.
        if sum(counts) != int(values.numel()):
            counts[-1] += int(values.numel()) - sum(counts)
    summary["histogram"] = {"edges": edges, "counts": counts}
    return summary


def _box_area_fraction(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_empty((0,), dtype=torch.float32)
    return boxes[:, 2].float().clamp_min(0.0) * boxes[:, 3].float().clamp_min(0.0)


def summarize_proposals(
    *,
    masks: torch.Tensor,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    image_hw: Sequence[int],
    small_area_threshold: float = 0.01,
    large_area_threshold: float = 0.25,
) -> Dict[str, Any]:
    """Return compact proposal statistics using normalized areas.

    ``mask_area`` is the foreground-pixel fraction of the image. ``box_area``
    is the normalized ``w*h`` value because SAM3 boxes are normalized.
    """

    height, width = int(image_hw[0]), int(image_hw[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"image_hw must be positive, got {tuple(image_hw)}")

    if masks.ndim == 4:
        masks = masks.squeeze(1)
    if masks.ndim != 3:
        raise ValueError(f"masks must have shape [N,H,W], got {tuple(masks.shape)}")
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"boxes must have shape [N,4], got {tuple(boxes.shape)}")
    if scores.ndim != 1:
        scores = scores.reshape(-1)
    count = int(scores.shape[0])
    if masks.shape[0] != count or boxes.shape[0] != count:
        raise ValueError("masks, boxes, and scores must contain the same number of proposals")

    mask_area = (masks.float().reshape(count, height * width).sum(dim=1) / float(height * width))
    box_area = _box_area_fraction(boxes)
    small_ratio = float((mask_area < float(small_area_threshold)).float().mean().item()) if count else 0.0
    large_ratio = float((mask_area >= float(large_area_threshold)).float().mean().item()) if count else 0.0
    score_summary = summarize_values(scores)
    mask_area_summary = summarize_values(mask_area)
    box_area_summary = summarize_values(box_area)
    return {
        "count": count,
        "score": score_summary,
        "mask_area": mask_area_summary,
        "box_area": box_area_summary,
        "score_histogram": score_summary["histogram"],
        "mask_area_histogram": mask_area_summary["histogram"],
        "small_region_ratio": small_ratio,
        "large_region_ratio": large_ratio,
        "small_area_threshold": float(small_area_threshold),
        "large_area_threshold": float(large_area_threshold),
    }


def _probe_label(spec: Mapping[str, Any]) -> str:
    probe_type = str(spec.get("type", "unknown"))
    if probe_type == "grid":
        return f"grid_{int(spec.get('grid_size', 0))}"
    if probe_type in {"horizontal_stripe", "vertical_stripe"}:
        return f"{probe_type}s"
    return probe_type


def _cxcywh_iou(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_xy1 = left[..., :2] - left[..., 2:] / 2.0
    left_xy2 = left[..., :2] + left[..., 2:] / 2.0
    right_xy1 = right[..., :2] - right[..., 2:] / 2.0
    right_xy2 = right[..., :2] + right[..., 2:] / 2.0
    intersection_xy1 = torch.maximum(left_xy1, right_xy1)
    intersection_xy2 = torch.minimum(left_xy2, right_xy2)
    intersection = (intersection_xy2 - intersection_xy1).clamp_min(0.0)
    intersection_area = intersection[..., 0] * intersection[..., 1]
    left_area = left[..., 2].clamp_min(0.0) * left[..., 3].clamp_min(0.0)
    right_area = right[..., 2].clamp_min(0.0) * right[..., 3].clamp_min(0.0)
    return intersection_area / (left_area + right_area - intersection_area).clamp_min(1e-6)


def attribute_proposals_to_probes(
    proposal_boxes: torch.Tensor,
    probe_specs: Sequence[Mapping[str, Any]],
) -> Dict[str, List[Any]]:
    """Assign each proposal box to its best spatially matching geometry probe.

    SAM3 currently returns one combined geometry output without prompt IDs.
    This is therefore an analysis attribution, not an exact per-prompt count.
    """

    if proposal_boxes.numel() == 0 or not probe_specs:
        return {"probe_type": [], "matched_probe_index": [], "matched": []}
    probes = torch.tensor([spec["box"] for spec in probe_specs], dtype=torch.float32)
    ious = _cxcywh_iou(proposal_boxes.detach().float().cpu()[:, None, :], probes[None, :, :])
    best_iou, best_index = ious.max(dim=1)
    return {
        "probe_type": [_probe_label(probe_specs[int(index)]) for index in best_index.tolist()],
        "matched_probe_index": [int(index) for index in best_index.tolist()],
        "matched": [bool(value > 0.0) for value in best_iou.tolist()],
        "best_iou": [float(value) for value in best_iou.tolist()],
    }


def _sum_graph_field(graph_batch: Mapping[str, Any], key: str) -> int:
    value = graph_batch.get(key)
    if value is None or not torch.is_tensor(value) or value.numel() == 0:
        return 0
    return int(value.detach().long().sum().item())


def summarize_graph_batch(graph_batch: Mapping[str, Any]) -> Dict[str, Any]:
    """Summarize graph size using the builder's existing diagnostics."""

    num_nodes = _sum_graph_field(graph_batch, "graph_num_nodes")
    topology_edges = _sum_graph_field(graph_batch, "graph_num_topology_edges")
    directional_edges = _sum_graph_field(graph_batch, "graph_num_directional_edges")
    total_edges = _sum_graph_field(graph_batch, "graph_num_total_edges")
    return {
        "num_nodes": num_nodes,
        "topology_edges": topology_edges,
        "directional_edges": directional_edges,
        "total_edges": total_edges,
        "average_degree": float(total_edges / max(num_nodes, 1)),
        "num_graphs": int(graph_batch.get("graph_num_nodes", torch.empty(0)).numel())
        if torch.is_tensor(graph_batch.get("graph_num_nodes"))
        else 0,
    }
