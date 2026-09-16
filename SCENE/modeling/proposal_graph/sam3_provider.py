from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import sys

import torch
from PIL import Image

from .proposal_analysis import attribute_proposals_to_probes, summarize_proposals


@dataclass(frozen=True)
class SAM3ProviderConfig:
    checkpoint_path: str
    bpe_path: str
    confidence_threshold: float
    max_proposals: int
    min_area: int
    default_prompt: str = "visual"
    prompt_mode: str = "visual_text"
    geometry_grid_sizes: Tuple[int, ...] = (1, 2, 4)
    geometry_include_stripes: bool = True
    geometry_stripe_count: int = 4
    geometry_include_center: bool = True
    geometry_nms_iou_threshold: float = 0.85
    geometry_box_scale: float = 1.15
    geometry_max_prompts: int = 32
    enable_proposal_analysis: bool = False


class SAM3ProposalProvider:
    SOURCE_VISUAL = 0
    SOURCE_GEOMETRY = 1

    def __init__(
        self,
        *,
        config: SAM3ProviderConfig,
        project_root: str,
        device: str,
        load_on_init: bool = False,
    ) -> None:
        self.config = config
        self.project_root = Path(project_root).resolve()
        self.device = device
        module_root = os.getenv("SEGEARTH_OV_ROOT")
        self._module_root = (
            Path(module_root).expanduser().resolve()
            if module_root
            else Path("./SegEarth-OV-3").resolve()
        )
        self._checkpoint_path = self._resolve_path(config.checkpoint_path)
        self._bpe_path = self._resolve_path(config.bpe_path)
        self._model = None
        self._processor = None

        self.validate_paths()
        if load_on_init:
            self.load()

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path

    @property
    def bpe_path(self) -> Path:
        return self._bpe_path

    @property
    def is_loaded(self) -> bool:
        return self._processor is not None

    def _resolve_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path)
        if candidate.is_absolute():
            return candidate
        return (self.project_root / candidate).resolve()

    def validate_paths(self) -> None:
        if not self._module_root.exists():
            raise FileNotFoundError(
                f"SegEarth-OV-3 module root not found at {self._module_root}."
            )
        if not self._checkpoint_path.is_file():
            raise FileNotFoundError(
                f"SAM3 checkpoint not found at {self._checkpoint_path}. "
                "Update MODEL.SEM_SEG_HEAD.SAM3_CHECKPOINT_PATH."
            )
        if not self._bpe_path.is_file():
            raise FileNotFoundError(
                f"SAM3 BPE vocabulary not found at {self._bpe_path}. "
                "Update MODEL.SEM_SEG_HEAD.SAM3_BPE_PATH."
            )

    def _ensure_import_path(self) -> None:
        module_root = str(self._module_root)
        if module_root not in sys.path:
            sys.path.insert(0, module_root)

    def load(self) -> None:
        if self.is_loaded:
            return

        self._ensure_import_path()
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        model = build_sam3_image_model(
            bpe_path=str(self._bpe_path),
            checkpoint_path=str(self._checkpoint_path),
            load_from_HF=False,
            device=self.device,
            eval_mode=True,
        )
        target_device = torch.device(self.device)
        model.to(device=target_device)
        # Some SAM3 builds keep the ViT trunk behind nested wrappers; move the
        # known image-backbone path defensively so image tensors and conv weights
        # cannot split across CUDA/CPU.
        trunk = getattr(getattr(getattr(model, "backbone", None), "vision_backbone", None), "trunk", None)
        if trunk is not None and hasattr(trunk, "to"):
            trunk.to(device=target_device)
        first_param = next(model.parameters(), None)
        if first_param is not None and first_param.device != target_device:
            raise RuntimeError(
                f"SAM3 model stayed on {first_param.device} after loading; expected {target_device}."
            )
        if trunk is not None:
            trunk_param = next(trunk.parameters(), None)
            if trunk_param is not None and trunk_param.device != target_device:
                raise RuntimeError(
                    f"SAM3 vision trunk stayed on {trunk_param.device} after loading; expected {target_device}."
                )
        self._model = model
        self._processor = Sam3Processor(
            model,
            confidence_threshold=self.config.confidence_threshold,
            device=self.device,
        )

    def _ensure_loaded(self) -> None:
        if not self.is_loaded:
            self.load()

    def _sync_for_analysis(self) -> None:
        if self.config.enable_proposal_analysis and self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def _summarize_state(self, state: Dict[str, Any], image_hw: Sequence[int]) -> Dict[str, Any]:
        return summarize_proposals(
            masks=state["masks"],
            boxes=state["boxes"],
            scores=state["object_score"],
            image_hw=image_hw,
        )

    def _filter_state(
        self,
        state: Dict[str, Any],
        *,
        source_id: int,
        image_hw: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        masks = state["masks"]
        mask_logits = state["masks_logits"]
        boxes = state["boxes"]
        scores = state["object_score"]
        device = mask_logits.device

        keep = torch.ones(mask_logits.shape[0], dtype=torch.bool, device=device)
        after_min_area_keep = keep.clone()
        after_nms_keep = keep.clone()
        if mask_logits.shape[0] > 0:
            areas = masks.flatten(1).sum(dim=1)
            keep &= areas >= int(self.config.min_area)
            after_min_area_keep = keep.clone()
            if masks.shape[0] > 0 and self.config.geometry_nms_iou_threshold < 1.0:
                keep_indices = torch.nonzero(keep, as_tuple=False).squeeze(1)
                if keep_indices.numel() > 0:
                    nms_keep = self._mask_nms(
                        masks[keep_indices],
                        scores[keep_indices],
                        iou_threshold=float(self.config.geometry_nms_iou_threshold),
                    )
                    nms_indices = keep_indices[nms_keep]
                    keep = torch.zeros_like(keep)
                    keep[nms_indices] = True
            after_nms_keep = keep.clone()
            if int(self.config.max_proposals) > 0 and keep.sum() > int(self.config.max_proposals):
                kept_indices = torch.nonzero(keep, as_tuple=False).squeeze(1)
                kept_scores = scores[kept_indices]
                topk = min(int(self.config.max_proposals), kept_scores.shape[0])
                topk_idx = kept_indices[torch.topk(kept_scores, k=topk).indices]
                keep = torch.zeros_like(keep)
                keep[topk_idx] = True
        else:
            areas = torch.zeros(0, device=device)

        feature_pyramid = state["backbone_out"].get("backbone_fpn", [])
        vision_features = state["backbone_out"].get("vision_features")

        filtered = {
            "masks": masks[keep],
            "mask_logits": mask_logits[keep],
            "boxes": boxes[keep],
            "scores": scores[keep],
            "areas": areas[keep] if areas.numel() else areas,
            "proposal_source": torch.full(
                (int(keep.sum().item()),),
                int(source_id),
                dtype=torch.long,
                device=device,
            ),
            "semantic_mask_logits": state["semantic_mask_logits"],
            "presence_score": state["presence_score"],
            "vision_features": vision_features,
            "feature_pyramid": feature_pyramid,
        }
        if self.config.enable_proposal_analysis:
            if image_hw is None:
                image_hw = (int(masks.shape[-2]), int(masks.shape[-1]))
            stage_states = {
                "raw": {"masks": masks, "boxes": boxes, "scores": scores},
                "after_min_area": {
                    "masks": masks[after_min_area_keep],
                    "boxes": boxes[after_min_area_keep],
                    "scores": scores[after_min_area_keep],
                },
                "after_branch_nms": {
                    "masks": masks[after_nms_keep],
                    "boxes": boxes[after_nms_keep],
                    "scores": scores[after_nms_keep],
                },
                "final": {
                    "masks": filtered["masks"],
                    "boxes": filtered["boxes"],
                    "scores": filtered["scores"],
                },
            }
            filtered["proposal_analysis"] = {
                "raw": self._summarize_state(state, image_hw),
                "after_min_area": summarize_proposals(
                    masks=stage_states["after_min_area"]["masks"],
                    boxes=stage_states["after_min_area"]["boxes"],
                    scores=stage_states["after_min_area"]["scores"],
                    image_hw=image_hw,
                ),
                "after_branch_nms": summarize_proposals(
                    masks=stage_states["after_branch_nms"]["masks"],
                    boxes=stage_states["after_branch_nms"]["boxes"],
                    scores=stage_states["after_branch_nms"]["scores"],
                    image_hw=image_hw,
                ),
                "filtered": summarize_proposals(
                    masks=filtered["masks"],
                    boxes=filtered["boxes"],
                    scores=filtered["scores"],
                    image_hw=image_hw,
                ),
                "branch_nms_retention_ratio": float(
                    stage_states["after_branch_nms"]["scores"].shape[0]
                    / max(stage_states["after_min_area"]["scores"].shape[0], 1)
                ),
            }
            filtered["_analysis_stage_states"] = stage_states
        return filtered

    def _merge_filtered_states(
        self,
        states: Sequence[Dict[str, Any]],
        *,
        image_hw: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        valid_states = [state for state in states if state["masks"].shape[0] > 0]
        template = states[0]
        if not valid_states:
            merged = dict(template)
            merged.pop("_analysis_stage_states", None)
            for key in ("masks", "mask_logits", "boxes", "scores", "areas", "proposal_source"):
                merged[key] = template[key][:0]
            if self.config.enable_proposal_analysis:
                merged["proposal_analysis"] = {
                    "merge_before_global_nms": {"count": 0},
                    "global_nms": {"count": 0},
                    "source_counts_before_global_nms": {},
                    "source_counts_after_global_nms": {},
                    "source_retention_ratio": {},
                }
            return merged

        merged = dict(template)
        merged.pop("_analysis_stage_states", None)
        for key in ("masks", "mask_logits", "boxes", "scores", "areas", "proposal_source"):
            merged[key] = torch.cat([state[key] for state in valid_states], dim=0)

        if self.config.enable_proposal_analysis and image_hw is None:
            image_hw = (int(merged["masks"].shape[-2]), int(merged["masks"].shape[-1]))
        analysis_before_global = None
        if self.config.enable_proposal_analysis:
            analysis_before_global = summarize_proposals(
                masks=merged["masks"],
                boxes=merged["boxes"],
                scores=merged["scores"],
                image_hw=image_hw,
            )
        source_counts_before = {
            str(int(source_id)): int((merged["proposal_source"] == source_id).sum().item())
            for source_id in torch.unique(merged["proposal_source"]).tolist()
        }

        keep = torch.ones(merged["scores"].shape[0], dtype=torch.bool, device=merged["scores"].device)
        if merged["masks"].shape[0] > 0 and self.config.geometry_nms_iou_threshold < 1.0:
            nms_keep = self._mask_nms(
                merged["masks"],
                merged["scores"],
                iou_threshold=float(self.config.geometry_nms_iou_threshold),
            )
            keep = torch.zeros_like(keep)
            keep[nms_keep] = True

        if int(self.config.max_proposals) > 0 and keep.sum() > int(self.config.max_proposals):
            global_nms_keep = keep.clone()
            kept_indices = torch.nonzero(keep, as_tuple=False).squeeze(1)
            kept_scores = merged["scores"][kept_indices]
            topk = min(int(self.config.max_proposals), kept_scores.shape[0])
            topk_idx = kept_indices[torch.topk(kept_scores, k=topk).indices]
            keep = torch.zeros_like(keep)
            keep[topk_idx] = True
        else:
            global_nms_keep = keep.clone()

        source_counts_after_global_nms = {}
        analysis_after_global_nms = None
        if self.config.enable_proposal_analysis:
            analysis_after_global_nms = summarize_proposals(
                masks=merged["masks"][global_nms_keep],
                boxes=merged["boxes"][global_nms_keep],
                scores=merged["scores"][global_nms_keep],
                image_hw=image_hw,
            )
            source_counts_after_global_nms = {
                str(int(source_id)): int(
                    (merged["proposal_source"][global_nms_keep] == source_id).sum().item()
                )
                for source_id in torch.unique(merged["proposal_source"][global_nms_keep]).tolist()
            }

        for key in ("masks", "mask_logits", "boxes", "scores", "areas", "proposal_source"):
            merged[key] = merged[key][keep]
        if self.config.enable_proposal_analysis:
            analysis_final = summarize_proposals(
                masks=merged["masks"],
                boxes=merged["boxes"],
                scores=merged["scores"],
                image_hw=image_hw,
            )
            source_counts_final = {
                str(int(source_id)): int((merged["proposal_source"] == source_id).sum().item())
                for source_id in torch.unique(merged["proposal_source"]).tolist()
            }
            source_retention = {
                source: float(source_counts_after_global_nms.get(source, 0) / max(count, 1))
                for source, count in source_counts_before.items()
            }
            source_final_retention = {
                source: float(source_counts_final.get(source, 0) / max(count, 1))
                for source, count in source_counts_before.items()
            }
            merged["proposal_analysis"] = {
                "merge_before_global_nms": analysis_before_global,
                "global_nms": analysis_after_global_nms,
                "source_counts_before_global_nms": source_counts_before,
                "source_counts_after_global_nms": source_counts_after_global_nms,
                "final_after_topk": analysis_final,
                "source_counts_final": source_counts_final,
                "source_retention_ratio": source_retention,
                "source_final_retention_ratio": source_final_retention,
            }
        return merged

    def _mask_nms(self, masks: torch.Tensor, scores: torch.Tensor, *, iou_threshold: float) -> torch.Tensor:
        if masks.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=scores.device)
        flat = masks.flatten(1).float()
        areas = flat.sum(dim=1).clamp_min(1e-6)
        order = scores.argsort(descending=True)
        selected: List[torch.Tensor] = []
        while order.numel() > 0:
            current = order[0]
            selected.append(current)
            if order.numel() == 1:
                break
            rest = order[1:]
            inter = (flat[current].unsqueeze(0) * flat[rest]).sum(dim=1)
            union = areas[current] + areas[rest] - inter
            iou = inter / union.clamp_min(1e-6)
            order = rest[iou <= iou_threshold]
        return torch.stack(selected, dim=0) if selected else torch.empty((0,), dtype=torch.long, device=scores.device)

    def _generate_geometry_probe_specs(self) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []
        scale = max(float(self.config.geometry_box_scale), 1e-3)
        for grid_size in self.config.geometry_grid_sizes:
            grid = max(int(grid_size), 1)
            box_size = min(scale / float(grid), 1.0)
            for y_idx in range(grid):
                for x_idx in range(grid):
                    specs.append({
                        "type": "grid",
                        "grid_size": grid,
                        "index": len(specs),
                        "box": [
                            (x_idx + 0.5) / float(grid),
                            (y_idx + 0.5) / float(grid),
                            box_size,
                            box_size,
                        ],
                    })

        if self.config.geometry_include_stripes:
            stripe_count = max(int(self.config.geometry_stripe_count), 1)
            stripe_size = min(scale / float(stripe_count), 1.0)
            for idx in range(stripe_count):
                center = (idx + 0.5) / float(stripe_count)
                specs.append({
                    "type": "horizontal_stripe",
                    "index": idx,
                    "box": [0.5, center, 1.0, stripe_size],
                })
                specs.append({
                    "type": "vertical_stripe",
                    "index": idx,
                    "box": [center, 0.5, stripe_size, 1.0],
                })

        if self.config.geometry_include_center:
            specs.append({
                "type": "center",
                "index": 0,
                "box": [0.5, 0.5, 0.5, 0.5],
            })

        max_prompts = int(self.config.geometry_max_prompts)
        return specs[:max_prompts] if max_prompts > 0 else specs

    def _generate_geometry_boxes(self) -> List[List[float]]:
        return [spec["box"] for spec in self._generate_geometry_probe_specs()]

    def _build_probe_analysis(
        self,
        raw_state: Dict[str, Any],
        filtered_state: Dict[str, Any],
        probe_specs: Sequence[Dict[str, Any]],
        image_hw: Sequence[int],
        stage_states: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, Any]:
        if stage_states is None:
            stage_states = {
                "raw": {
                    "masks": raw_state["masks"],
                    "boxes": raw_state["boxes"],
                    "scores": raw_state["object_score"],
                },
                "final": {
                    "masks": filtered_state["masks"],
                    "boxes": filtered_state["boxes"],
                    "scores": filtered_state["scores"],
                },
            }
        raw_assignment = attribute_proposals_to_probes(raw_state["boxes"], probe_specs)
        filtered_assignment = attribute_proposals_to_probes(filtered_state["boxes"], probe_specs)
        labels = []
        for spec in probe_specs:
            probe_type = spec["type"]
            label = f"grid_{int(spec['grid_size'])}" if probe_type == "grid" else (
                f"{probe_type}s" if probe_type in {"horizontal_stripe", "vertical_stripe"} else probe_type
            )
            if label not in labels:
                labels.append(label)

        result: Dict[str, Any] = {
            "attribution_method": "proposal_box_to_probe_iou",
            "probe_count": len(probe_specs),
            "types": {},
        }
        for label in labels:
            raw_indices = [idx for idx, value in enumerate(raw_assignment["probe_type"]) if value == label]
            filtered_indices = [idx for idx, value in enumerate(filtered_assignment["probe_type"]) if value == label]
            raw_index_tensor = torch.tensor(raw_indices, dtype=torch.long, device=raw_state["masks"].device)
            filtered_index_tensor = torch.tensor(filtered_indices, dtype=torch.long, device=filtered_state["masks"].device)
            raw_summary = summarize_proposals(
                masks=raw_state["masks"][raw_index_tensor],
                boxes=raw_state["boxes"][raw_index_tensor],
                scores=raw_state["object_score"][raw_index_tensor],
                image_hw=image_hw,
            )
            filtered_summary = summarize_proposals(
                masks=filtered_state["masks"][filtered_index_tensor],
                boxes=filtered_state["boxes"][filtered_index_tensor],
                scores=filtered_state["scores"][filtered_index_tensor],
                image_hw=image_hw,
            )
            raw_count = int(raw_summary["count"])
            type_result = {
                "generated_proposal_count": raw_count,
                "effective_proposal_count": int(filtered_summary["count"]),
                "average_mask_area": filtered_summary["mask_area"]["mean"],
                "score": filtered_summary["score"],
                "raw": raw_summary,
                "filtered": filtered_summary,
                "filter_retention_ratio": float(filtered_summary["count"] / max(raw_count, 1)),
            }
            for stage_name, stage_state in stage_states.items():
                if stage_name == "raw":
                    assignment = raw_assignment
                else:
                    assignment = attribute_proposals_to_probes(stage_state["boxes"], probe_specs)
                stage_indices = [
                    idx for idx, value in enumerate(assignment["probe_type"]) if value == label
                ]
                index_tensor = torch.tensor(
                    stage_indices,
                    dtype=torch.long,
                    device=stage_state["masks"].device,
                )
                stage_summary = summarize_proposals(
                    masks=stage_state["masks"][index_tensor],
                    boxes=stage_state["boxes"][index_tensor],
                    scores=stage_state["scores"][index_tensor],
                    image_hw=image_hw,
                )
                type_result[stage_name] = stage_summary
            after_min_count = type_result.get("after_min_area", {}).get("count", raw_count)
            after_nms_count = type_result.get("after_branch_nms", {}).get("count", after_min_count)
            type_result["nms_retention_ratio"] = float(
                after_nms_count / max(after_min_count, 1)
            )
            result["types"][label] = type_result
        return result

    @torch.inference_mode()
    def _run_geometry_prompts(self, base_state: Dict[str, Any], boxes: Sequence[Sequence[float]]) -> Dict[str, Any]:
        assert self._processor is not None
        self._processor.reset_all_prompts(base_state)
        if "language_features" not in base_state["backbone_out"]:
            text_outputs = self._processor.model.backbone.forward_text(
                [self.config.default_prompt],
                device=self._processor.device,
            )
            base_state["backbone_out"].update(text_outputs)
        base_state["geometric_prompt"] = self._processor.model._get_dummy_prompt()
        if boxes:
            box_tensor = torch.tensor(boxes, device=self._processor.device, dtype=torch.float32).view(-1, 1, 4)
            labels = torch.ones((box_tensor.shape[0], 1), device=self._processor.device, dtype=torch.bool)
            base_state["geometric_prompt"].append_boxes(box_tensor, labels)
        return self._processor._forward_grounding(base_state)

    @torch.inference_mode()
    def generate(
        self,
        image: Image.Image,
        prompts: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        self._ensure_loaded()
        assert self._processor is not None

        analysis_enabled = bool(self.config.enable_proposal_analysis)
        analysis_start = None
        if analysis_enabled:
            self._sync_for_analysis()
            analysis_start = time.perf_counter()

        prompt_list: List[str] = list(prompts) if prompts else [self.config.default_prompt]
        prompt_outputs: List[Dict[str, Any]] = []

        base_state = self._processor.set_image(image)
        image_hw = (int(image.height), int(image.width))
        if self.config.prompt_mode == "auto_geometry":
            probe_specs = self._generate_geometry_probe_specs()
            geometry_boxes = [spec["box"] for spec in probe_specs]
            prompted_state = self._run_geometry_prompts(base_state, geometry_boxes)
            filtered_state = self._filter_state(
                prompted_state,
                source_id=self.SOURCE_GEOMETRY,
                image_hw=image_hw,
            )
            geometry_stage_states = filtered_state.pop("_analysis_stage_states", None)
            filtered_state["prompt"] = self.config.default_prompt
            filtered_state["prompt_mode"] = self.config.prompt_mode
            filtered_state["geometry_prompts"] = geometry_boxes
            filtered_state["prompt_outputs"] = [{
                "prompt": self.config.default_prompt,
                "prompt_mode": self.config.prompt_mode,
                "num_geometry_prompts": len(geometry_boxes),
                "num_proposals": int(filtered_state["scores"].shape[0]),
            }]
            if analysis_enabled:
                filtered_state["proposal_analysis"] = {
                    "enabled": True,
                    "prompt_mode": self.config.prompt_mode,
                    "geometry": filtered_state["proposal_analysis"],
                    "geometry_probe_types": self._build_probe_analysis(
                        prompted_state,
                        filtered_state,
                        probe_specs,
                        image_hw,
                        stage_states=geometry_stage_states,
                    ),
                    "latency_ms": {
                        "proposal_generation": (time.perf_counter() - analysis_start) * 1000.0,
                    },
                }
            return filtered_state

        if self.config.prompt_mode == "visual_geometry":
            self._processor.reset_all_prompts(base_state)
            visual_state = self._processor.set_text_prompt(prompt=self.config.default_prompt, state=base_state)
            visual_filtered = self._filter_state(
                visual_state,
                source_id=self.SOURCE_VISUAL,
                image_hw=image_hw,
            )

            probe_specs = self._generate_geometry_probe_specs()
            geometry_boxes = [spec["box"] for spec in probe_specs]
            geometry_state = self._run_geometry_prompts(base_state, geometry_boxes)
            geometry_filtered = self._filter_state(
                geometry_state,
                source_id=self.SOURCE_GEOMETRY,
                image_hw=image_hw,
            )
            geometry_stage_states = geometry_filtered.pop("_analysis_stage_states", None)

            filtered_state = self._merge_filtered_states(
                [visual_filtered, geometry_filtered],
                image_hw=image_hw,
            )
            filtered_state["prompt"] = self.config.default_prompt
            filtered_state["prompt_mode"] = self.config.prompt_mode
            filtered_state["geometry_prompts"] = geometry_boxes
            filtered_state["prompt_outputs"] = [
                {
                    "prompt": self.config.default_prompt,
                    "prompt_mode": "visual_text",
                    "num_proposals": int(visual_filtered["scores"].shape[0]),
                },
                {
                    "prompt": self.config.default_prompt,
                    "prompt_mode": "auto_geometry",
                    "num_geometry_prompts": len(geometry_boxes),
                    "num_proposals": int(geometry_filtered["scores"].shape[0]),
                },
            ]
            if analysis_enabled:
                filtered_state["proposal_analysis"] = {
                    "enabled": True,
                    "prompt_mode": self.config.prompt_mode,
                    "visual": visual_filtered["proposal_analysis"],
                    "geometry": geometry_filtered["proposal_analysis"],
                    "geometry_probe_types": self._build_probe_analysis(
                        geometry_state,
                        geometry_filtered,
                        probe_specs,
                        image_hw,
                        stage_states=geometry_stage_states,
                    ),
                    "merge": filtered_state["proposal_analysis"],
                    "latency_ms": {
                        "proposal_generation": (time.perf_counter() - analysis_start) * 1000.0,
                    },
                }
            return filtered_state

        for prompt in prompt_list:
            self._processor.reset_all_prompts(base_state)
            prompted_state = self._processor.set_text_prompt(prompt=prompt, state=base_state)
            filtered_state = self._filter_state(
                prompted_state,
                source_id=self.SOURCE_VISUAL,
                image_hw=image_hw,
            )
            filtered_state.pop("_analysis_stage_states", None)
            filtered_state["prompt"] = prompt
            filtered_state["prompt_mode"] = self.config.prompt_mode
            prompt_outputs.append(filtered_state)

        primary = prompt_outputs[0]
        primary["prompt_outputs"] = prompt_outputs
        if analysis_enabled:
            primary["proposal_analysis"] = {
                "enabled": True,
                "prompt_mode": self.config.prompt_mode,
                "visual": primary["proposal_analysis"],
                "latency_ms": {
                    "proposal_generation": (time.perf_counter() - analysis_start) * 1000.0,
                },
            }
        return primary
