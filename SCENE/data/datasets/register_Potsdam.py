import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

POTSDAM_CLASSES = [
    "impervious surface",
    "buildings",
    "low vegetation",
    "tree",
    "car",
]


def _get_potsdam_meta():
    return {
        "stuff_classes": POTSDAM_CLASSES,
    }


def register_Potsdam(root):
    meta = _get_potsdam_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("all", "Potsdam/img", "Potsdam/ann"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        dataset_name = f"Potsdam_{name}_sem_seg"
        DatasetCatalog.register(
            dataset_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="png"),
        )
        MetadataCatalog.get(dataset_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=5,
            **meta,
        )


root = "./datasets/test"
register_Potsdam(root)
