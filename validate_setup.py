import argparse

import torch
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config

from SCENE import add_scene_config, prepare_scene_cfg, validate_scene_cfg


def build_parser():
    parser = argparse.ArgumentParser(description="Validate SCENE config and model bootstrap.")
    parser.add_argument("--config-file", required=True, help="Path to config yaml.")
    parser.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        default=[],
        help="Override config options from the command line.",
    )
    parser.add_argument(
        "--skip-model-build",
        action="store_true",
        help="Only validate config paths without constructing the model.",
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
    checked_paths = validate_scene_cfg(cfg, require_model_weights=bool(cfg.MODEL.WEIGHTS))
    return cfg, checked_paths


def main():
    args = build_parser().parse_args()
    cfg, checked_paths = setup_cfg(args)
    print("validated_paths", checked_paths)
    if args.skip_model_build:
        return

    torch.set_float32_matmul_precision("high")
    model = build_model(cfg)
    print(
        "model_ready",
        {
            "meta_arch": cfg.MODEL.META_ARCHITECTURE,
            "head_name": cfg.MODEL.SEM_SEG_HEAD.NAME,
            "use_sam3": bool(cfg.MODEL.SEM_SEG_HEAD.USE_SAM3_PROPOSAL),
            "graph_mode": cfg.MODEL.SEM_SEG_HEAD.GRAPH_MODE,
            "use_node_text_match": bool(cfg.MODEL.SEM_SEG_HEAD.USE_NODE_TEXT_MATCH),
            "use_remote_fc_decoder": bool(cfg.MODEL.SEM_SEG_HEAD.USE_REMOTE_FC_DECODER),
            "model_type": type(model).__name__,
        },
    )


if __name__ == "__main__":
    main()
