# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import copy
import itertools
import logging
import os
import time
from collections import OrderedDict
from datetime import timedelta
from typing import Any, Dict, List, Set
# from memory_profiler import profile
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.engine import defaults as d2_engine_defaults
from detectron2.engine.train_loop import AMPTrainer, SimpleTrainer
from detectron2.evaluation import CityscapesInstanceEvaluator, CityscapesSemSegEvaluator, \
    COCOEvaluator, COCOPanopticEvaluator, DatasetEvaluators, SemSegEvaluator, verify_results, \
    DatasetEvaluator

from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from detectron2.utils.events import EventWriter, get_event_storage

from detectron2.utils.file_io import PathManager
import numpy as np
from PIL import Image
import glob

import pycocotools.mask as mask_util

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize

# from detectron2.evaluation import SemSegGzeroEvaluator
# from mask_former.evaluation.sem_seg_evaluation_gzero import SemSegGzeroEvaluator

class VOCbEvaluator(SemSegEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """
    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=np.int)
            pred[pred >= 20] = 20
            with PathManager.open(self.input_file_to_gt_file[input["file_name"]], "rb") as f:
                gt = np.array(Image.open(f), dtype=np.int)

            gt[gt == self._ignore_label] = self._num_classes

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))


class OpenVocabSemSegEvaluator(SemSegEvaluator):
    def __init__(self, dataset_name, cfg, distributed=True, output_dir=None, **kwargs):
        super().__init__(dataset_name, distributed=distributed, output_dir=output_dir, **kwargs)
        self._cfg = cfg
        self._seen_indices, self._unseen_indices = self._build_split_indices()

    def _normalize_class_name(self, name):
        return " ".join(str(name).strip().lower().split())

    def _metadata_class_names(self, dataset_name):
        metadata = MetadataCatalog.get(dataset_name)
        class_names = getattr(metadata, "stuff_classes", None)
        if not class_names:
            raise ValueError(
                f"Dataset {dataset_name!r} has no non-empty 'stuff_classes' metadata; "
                "cannot construct open-vocabulary Seen/Unseen metrics."
            )
        return list(class_names)

    def _training_class_names(self):
        train_dataset_names = tuple(self._cfg.DATASETS.TRAIN)
        if not train_dataset_names:
            raise ValueError(
                "cfg.DATASETS.TRAIN is empty; cannot determine the training vocabulary "
                "for open-vocabulary Seen/Unseen metrics."
            )

        class_names = []
        seen_names = set()
        for dataset_name in train_dataset_names:
            for class_name in self._metadata_class_names(dataset_name):
                normalized_name = self._normalize_class_name(class_name)
                if normalized_name not in seen_names:
                    seen_names.add(normalized_name)
                    class_names.append(class_name)
        return class_names

    def _map_names_to_indices(self, class_names):
        if not class_names:
            return []
        name_to_index = {
            self._normalize_class_name(name): idx for idx, name in enumerate(self._class_names)
        }
        indices = []
        for class_name in class_names:
            index = name_to_index.get(self._normalize_class_name(class_name))
            if index is not None:
                indices.append(index)
        return sorted(set(indices))

    def _build_split_indices(self):
        train_classes = self._training_class_names()
        test_classes = self._metadata_class_names(self._dataset_name)

        background_name = "background"
        train_norm = {
            self._normalize_class_name(name)
            for name in train_classes
            if self._normalize_class_name(name) != background_name
        }
        test_classes_without_background = [
            name
            for name in test_classes
            if self._normalize_class_name(name) != background_name
        ]
        seen_class_names = [
            name
            for name in test_classes_without_background
            if self._normalize_class_name(name) in train_norm
        ]
        unseen_class_names = [
            name
            for name in test_classes_without_background
            if self._normalize_class_name(name) not in train_norm
        ]

        seen_indices = self._map_names_to_indices(seen_class_names)
        unseen_indices = self._map_names_to_indices(unseen_class_names)

        if not seen_indices or not unseen_indices:
            raise ValueError(
                f"Invalid open-vocabulary split for {self._dataset_name!r}: "
                f"seen={len(seen_indices)}, unseen={len(unseen_indices)}. "
                "Ensure cfg.DATASETS.TRAIN points to the training vocabulary and "
                "the test dataset metadata contains both seen and unseen classes."
            )

        ignored_classes = [
            name for name in test_classes if self._normalize_class_name(name) == background_name
        ]
        self._logger.info("========== Open Vocabulary Split (%s) ==========", self._dataset_name)
        self._logger.info("Train classes: %s", train_classes)
        self._logger.info("Seen classes: %s", seen_class_names)
        self._logger.info("Unseen classes: %s", unseen_class_names)
        self._logger.info("Ignored classes: %s", ignored_classes or ["<none>"])
        self._logger.info("===============================================")

        return seen_indices, unseen_indices

    def _compute_split_miou(self, conf_matrix, class_indices):
        if not class_indices:
            return float("nan"), 0, 0
        tp = conf_matrix.diagonal()[:-1].astype(float)
        pos_gt = np.sum(conf_matrix[:-1, :-1], axis=0).astype(float)
        pos_pred = np.sum(conf_matrix[:-1, :-1], axis=1).astype(float)
        union = pos_gt + pos_pred - tp
        valid = np.logical_and(pos_gt > 0, union > 0)
        class_indices = np.asarray(class_indices, dtype=np.int64)
        valid_indices = class_indices[valid[class_indices]]
        if valid_indices.size == 0:
            return float("nan"), 0, int(class_indices.size)
        iou = tp[valid_indices] / union[valid_indices]
        return float(np.mean(iou)), int(valid_indices.size), int(class_indices.size)

    def evaluate(self):
        results = super().evaluate()
        if results is None:
            return results

        seen_miou, valid_seen_count, total_seen_count = self._compute_split_miou(
            self._conf_matrix,
            self._seen_indices,
        )
        unseen_miou, valid_unseen_count, total_unseen_count = self._compute_split_miou(
            self._conf_matrix,
            self._unseen_indices,
        )

        self._logger.info(
            "Open-vocabulary class coverage for %s: Seen %d/%d, Unseen %d/%d",
            self._dataset_name,
            valid_seen_count,
            total_seen_count,
            valid_unseen_count,
            total_unseen_count,
        )

        if np.isfinite(seen_miou):
            results["sem_seg"]["Seen-mIoU"] = 100.0 * seen_miou
        if np.isfinite(unseen_miou):
            results["sem_seg"]["Unseen-mIoU"] = 100.0 * unseen_miou
        if np.isfinite(seen_miou) and np.isfinite(unseen_miou) and (seen_miou + unseen_miou) > 0:
            results["sem_seg"]["HM"] = 100.0 * (2.0 * seen_miou * unseen_miou / (seen_miou + unseen_miou))

        if any(metric in results["sem_seg"] for metric in ("Seen-mIoU", "Unseen-mIoU", "HM")):
            self._logger.info("Open-vocabulary split metrics: %s", results["sem_seg"])
            if self._output_dir:
                file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
                with PathManager.open(file_path, "wb") as f:
                    torch.save(results["sem_seg"], f)

        return results


class IterationProgressWriter(EventWriter):
    def __init__(self, max_iter: int) -> None:
        self.max_iter = max_iter
        self.logger = logging.getLogger("detectron2")
        self._last_write = -1
        self._last_time = time.perf_counter()

    def write(self) -> None:
        storage = get_event_storage()
        if storage.iter == self._last_write:
            return

        current_iter = storage.iter + 1
        percent = 100.0 * current_iter / max(self.max_iter, 1)
        latest_metrics = storage.latest()
        metric_names = (
            "total_loss",
            "loss_sem_seg",
            "loss_vl",
            "loss_node",
            "loss_hard_refined",
            "loss_hard_original",
            "loss_node_hard",
            "loss_node_pseudo",
            "loss_node_consistency",
            "num_node_hard",
            "num_node_pseudo",
            "context_bias_mean",
            "context_bias_std",
            "context_bias_max",
            "original_node_logits_mean",
            "original_node_logits_std",
            "node_logits_mean",
            "node_logits_std",
            "lr",
        )
        metric_parts = []
        for name in metric_names:
            latest_entry = latest_metrics.get(name)
            if latest_entry is None:
                continue
            value, value_iter = latest_entry
            if value_iter != storage.iter:
                continue
            if isinstance(value, float):
                metric_parts.append(f"{name}={value:.4f}")
            else:
                metric_parts.append(f"{name}={value}")

        now = time.perf_counter()
        iter_time = now - self._last_time
        self._last_time = now
        if current_iter > 1:
            metric_parts.append(f"iter_time={iter_time:.2f}s")

        self.logger.info(
            "progress %d/%d (%.2f%%) %s",
            current_iter,
            self.max_iter,
            percent,
            " ".join(metric_parts),
        )
        self._last_write = storage.iter


class DiagnosticTrainerMixin:
    def _set_model_training_progress(self) -> None:
        data_loader_len = len(self.data_loader) if hasattr(self.data_loader, "__len__") else 0
        current_epoch = self.iter // max(int(data_loader_len), 1)
        current_iter = self.iter
        for model_obj in (self.model, getattr(self.model, "module", None)):
            if model_obj is None:
                continue
            setattr(model_obj, "current_iter", current_iter)
            setattr(model_obj, "current_epoch", current_epoch)

    @staticmethod
    def _split_loss_dict(loss_dict: Dict[str, torch.Tensor]):
        optimize_dict = {}
        metric_dict = {}
        for key, value in loss_dict.items():
            if key.startswith("metric_"):
                metric_dict[key[len("metric_"):]] = value
            else:
                optimize_dict[key] = value
        return optimize_dict, metric_dict

    @staticmethod
    def write_metrics(
        optimize_dict: Dict[str, torch.Tensor],
        metric_dict: Dict[str, torch.Tensor],
        data_time: float,
        cur_iter: int,
        prefix: str = "",
    ) -> None:
        optimize_metrics = {k: v.detach().cpu().item() for k, v in optimize_dict.items()}
        extra_metrics = {k: v.detach().cpu().item() for k, v in metric_dict.items()}
        metrics_dict = {**optimize_metrics, **extra_metrics, "data_time": data_time}

        storage = get_event_storage()
        storage.put_scalar("rank_data_time", data_time, cur_iter=cur_iter)

        all_metrics_dict = comm.gather(metrics_dict)
        if comm.is_main_process():
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time, cur_iter=cur_iter)

            common_metric_keys = set(all_metrics_dict[0].keys())
            for rank_metrics in all_metrics_dict[1:]:
                common_metric_keys.intersection_update(rank_metrics.keys())

            reduced_metrics = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in sorted(common_metric_keys)
            }
            total_losses_reduced = sum(
                reduced_metrics[k] for k in optimize_metrics.keys() if k in reduced_metrics
            )
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={cur_iter}!\n"
                    f"loss_dict = {reduced_metrics}"
                )

            storage.put_scalar(f"{prefix}total_loss", total_losses_reduced, cur_iter=cur_iter)
            if reduced_metrics:
                storage.put_scalars(cur_iter=cur_iter, **reduced_metrics)

    def _write_metrics(
        self,
        optimize_dict: Dict[str, torch.Tensor],
        metric_dict: Dict[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
        iter: int = None,
    ) -> None:
        if iter is None:
            iter = self.iter
        self.write_metrics(optimize_dict, metric_dict, data_time, iter, prefix)


class DiagnosticSimpleTrainer(DiagnosticTrainerMixin, SimpleTrainer):
    def run_step(self):
        assert self.model.training, "[DiagnosticSimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        if self.zero_grad_before_forward:
            self.optimizer.zero_grad()

        self._set_model_training_progress()
        raw_loss_dict = self.model(data)
        if isinstance(raw_loss_dict, torch.Tensor):
            losses = raw_loss_dict
            optimize_dict = {"total_loss": raw_loss_dict}
            metric_dict = {}
        else:
            optimize_dict, metric_dict = self._split_loss_dict(raw_loss_dict)
            losses = sum(optimize_dict.values())

        if not self.zero_grad_before_forward:
            self.optimizer.zero_grad()
        losses.backward()

        self.after_backward()

        if self.async_write_metrics:
            self.concurrent_executor.submit(
                self._write_metrics, optimize_dict, metric_dict, data_time, iter=self.iter
            )
        else:
            self._write_metrics(optimize_dict, metric_dict, data_time)

        self.optimizer.step()


class DiagnosticAMPTrainer(DiagnosticTrainerMixin, AMPTrainer):
    def run_step(self):
        assert self.model.training, "[DiagnosticAMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[DiagnosticAMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        if self.zero_grad_before_forward:
            self.optimizer.zero_grad()
        self._set_model_training_progress()
        with autocast(dtype=self.precision):
            raw_loss_dict = self.model(data)
            if isinstance(raw_loss_dict, torch.Tensor):
                losses = raw_loss_dict
                optimize_dict = {"total_loss": raw_loss_dict}
                metric_dict = {}
            else:
                optimize_dict, metric_dict = self._split_loss_dict(raw_loss_dict)
                losses = sum(optimize_dict.values())

        if not self.zero_grad_before_forward:
            self.optimizer.zero_grad()

        self.grad_scaler.scale(losses).backward()

        if self.log_grad_scaler:
            storage = get_event_storage()
            storage.put_scalar("[metric]grad_scaler", self.grad_scaler.get_scale())

        self.after_backward()

        if self.async_write_metrics:
            self.concurrent_executor.submit(
                self._write_metrics, optimize_dict, metric_dict, data_time, iter=self.iter
            )
        else:
            self._write_metrics(optimize_dict, metric_dict, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

# SCENE
from SCENE import (
    DETRPanopticDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_scene_config,
    prepare_scene_cfg,
    validate_scene_cfg,
)


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to DETR.
    """

    def __init__(self, cfg):
        original_create_ddp_model = d2_engine_defaults.create_ddp_model

        def create_ddp_model_with_unused_parameters(model, **kwargs):
            kwargs.setdefault("find_unused_parameters", True)
            return original_create_ddp_model(model, **kwargs)

        d2_engine_defaults.create_ddp_model = create_ddp_model_with_unused_parameters
        try:
            super().__init__(cfg)
        finally:
            d2_engine_defaults.create_ddp_model = original_create_ddp_model

        base_trainer = self._trainer
        if cfg.SOLVER.AMP.ENABLED:
            self._trainer = DiagnosticAMPTrainer(
                base_trainer.model,
                base_trainer.data_loader,
                base_trainer.optimizer,
                gather_metric_period=base_trainer.gather_metric_period,
                zero_grad_before_forward=base_trainer.zero_grad_before_forward,
                grad_scaler=base_trainer.grad_scaler,
                precision=base_trainer.precision,
                log_grad_scaler=base_trainer.log_grad_scaler,
                async_write_metrics=base_trainer.async_write_metrics,
            )
        else:
            self._trainer = DiagnosticSimpleTrainer(
                base_trainer.model,
                base_trainer.data_loader,
                base_trainer.optimizer,
                gather_metric_period=base_trainer.gather_metric_period,
                zero_grad_before_forward=base_trainer.zero_grad_before_forward,
                async_write_metrics=base_trainer.async_write_metrics,
            )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                OpenVocabSemSegEvaluator(
                    dataset_name,
                    cfg,
                    distributed=True,
                    output_dir=output_folder,
                )
            )

        if evaluator_type == "sem_seg_background":
            evaluator_list.append(
                VOCbEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
        ]:
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
        # DETR-style dataset mapper for COCO panoptic segmentation
        elif cfg.INPUT.DATASET_MAPPER_NAME == "detr_panoptic":
            mapper = DETRPanopticDatasetMapper(cfg, True)
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        # import ipdb;
        # ipdb.set_trace()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if "clip_model" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.CLIP_MULTIPLIER
                # for deformable detr

                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    def build_hooks(self):
        ret = super().build_hooks()
        if self.cfg.TEST.EVAL_PERIOD == 0:
            # Detectron2 otherwise evaluates at training end even with period=0.
            ret = [hook for hook in ret if not isinstance(hook, hooks.EvalHook)]
        for idx, hook in enumerate(ret):
            if isinstance(hook, hooks.PeriodicWriter):
                ret[idx] = hooks.PeriodicWriter(
                    self.build_writers(),
                    period=self.cfg.LOG_PERIOD,
                )
        return ret

    def build_writers(self):
        writers = super().build_writers()
        if self.cfg.LOG_PERIOD <= 1:
            filtered_writers = []
            for writer in writers:
                if writer.__class__.__name__ == "CommonMetricPrinter":
                    continue
                filtered_writers.append(writer)
            return [IterationProgressWriter(self.max_iter)] + filtered_writers
        return writers

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_scene_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    prepare_scene_cfg(cfg)
    validate_scene_cfg(cfg, require_model_weights=args.eval_only)
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask_former")
    return cfg


def main(args):
    cfg = setup(args)
    torch.set_float32_matmul_precision("high")
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)#恢复断点训练
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch_timeout = timedelta(hours=6) if args.eval_only else timedelta(minutes=30)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
        timeout=launch_timeout,
    )
