import argparse
import os
from pathlib import Path

import torch
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from SCENE import add_scene_config, prepare_scene_cfg, validate_scene_cfg
from demo.predictor import VisualizationDemo


def build_parser():
    parser = argparse.ArgumentParser(description="Batch export semantic segmentation predictions.")
    parser.add_argument("--config-file", required=True, help="Path to config yaml.")
    parser.add_argument("--output", required=True, help="Directory to save exported predictions.")
    parser.add_argument("--dataset", help="Registered dataset name to export.")
    parser.add_argument("--input-dir", help="Directory of input images to export.")
    parser.add_argument(
        "--glob",
        default="*.png",
        help="Glob pattern used with --input-dir. Default: *.png",
    )
    parser.add_argument(
        "--save-overlay",
        action="store_true",
        help="Save overlay images in addition to mask outputs.",
    )
    parser.add_argument(
        "--gt-from-dataset",
        action="store_true",
        help="When using --dataset, also export GT masks and compare canvases if sem_seg_file_name exists.",
    )
    parser.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional config overrides.",
    )
    return parser


def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_scene_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if not torch.cuda.is_available():
        cfg.defrost()
        cfg.MODEL.DEVICE = "cpu"
        cfg.freeze()
    prepare_scene_cfg(cfg)
    validate_scene_cfg(cfg, require_model_weights=bool(cfg.MODEL.WEIGHTS))
    return cfg


def iter_dataset_records(dataset_name):
    for record in DatasetCatalog.get(dataset_name):
        yield {
            "file_name": record["file_name"],
            "stem": Path(record["file_name"]).stem,
            "gt_file": record.get("sem_seg_file_name"),
        }


def iter_input_dir(input_dir, pattern):
    for path in sorted(Path(input_dir).glob(pattern)):
        if path.is_file():
            yield {
                "file_name": str(path),
                "stem": path.stem,
                "gt_file": None,
            }


def main():
    args = build_parser().parse_args()
    if bool(args.dataset) == bool(args.input_dir):
        raise ValueError("Specify exactly one of --dataset or --input-dir.")

    setup_logger(name="fvcore")
    logger = setup_logger()
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    if args.dataset:
        iterator = iter_dataset_records(args.dataset)
    else:
        iterator = iter_input_dir(args.input_dir, args.glob)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for item in iterator:
        image = read_image(item["file_name"], format="BGR")
        predictions, vis_output = demo.run_on_image(image)
        gt_mask = None
        if args.gt_from_dataset and item["gt_file"] and os.path.isfile(item["gt_file"]):
            gt_mask = read_image(item["gt_file"], format="L")

        if args.save_overlay and vis_output is not None:
            overlay_dir = output_dir / "detectron_overlay"
            overlay_dir.mkdir(parents=True, exist_ok=True)
            vis_output.save(overlay_dir / f"{item['stem']}.png")

        demo.save_visualizations(
            image_bgr=image,
            predictions=predictions,
            output_dir=output_dir,
            stem=item["stem"],
            gt_mask=gt_mask,
        )
        logger.info("exported %s", item["file_name"])


if __name__ == "__main__":
    main()
