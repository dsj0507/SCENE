import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

FLOODNET_CLASSES = [
    "Background",
    "building-flooded",
    "building-non-flooded",
    "road-flooded",
    "road-non-flooded",
    "water",
    "tree",
    "vehicle",
    "pool",
    "grass",
]


def _get_floodnet_meta():
    return {
        "stuff_classes": FLOODNET_CLASSES,
    }


def register_floodnet(root):
    meta = _get_floodnet_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("FloodNet", "FloodNet/val+test/img", "FloodNet/val+test/lbl"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        DatasetCatalog.register(
            name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg"),
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=0,
            **meta,
        )


root = "./datasets/test"
register_floodnet(root)
