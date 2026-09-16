import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

FLAIR_CLASSES = [
    "buildings",
    "pervious surface",
    "impervious surface",
    "bare soil",
    "water",
    "coniferous",
    "deciduous",
    "brushwood",
    "vineyard",
    "herbaceous vegetation",
    "agriculture land",
    "plowed land",
]


def _get_flair_meta():
    return {
        "stuff_classes": FLAIR_CLASSES,
    }


def register_flair(root):
    meta = _get_flair_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("FLAIR_test", "FLAIR/image", "FLAIR/mask"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        DatasetCatalog.register(
            name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="png"),
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=12,
            **meta,
        )


root = "./datasets/test"
register_flair(root)
