import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

FAST_CLASSES = [
    "A220",
    "A321",
    "A330",
    "A350",
    "ARJ21",
    "Baseball-Field",
    "Basketball-Court",
    "Boeing737",
    "Boeing747",
    "Boeing777",
    "Boeing787",
    "Bridge",
    "Bus",
    "C919",
    "Cargo-Truck",
    "Dry-Cargo-Ship",
    "Dump-Truck",
    "Engineering-Ship",
    "Excavator",
    "Fishing-Boat",
    "Football-Field",
    "Intersection",
    "Liquid-Cargo-Ship",
    "Motorboat",
    "other-airplane",
    "other-ship",
    "other-vehicle",
    "Passenger-Ship",
    "Roundabout",
    "Small-Car",
    "Tennis-Court",
    "Tractor",
    "Trailer",
    "Truck-Tractor",
    "Tugboat",
    "Van",
    "Warship",
]


def _get_fast_meta():
    return {
        "stuff_classes": FAST_CLASSES,
    }


def register_FAST(root):
    meta = _get_fast_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("FAST_val", "FAST/images", "FAST/masks"),
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
            ignore_label=255,
            **meta,
        )


root = "./datasets/test"
register_FAST(root)
