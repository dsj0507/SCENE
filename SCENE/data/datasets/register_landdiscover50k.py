from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from detectron2.data import DatasetCatalog, MetadataCatalog


DATA_ROOT = Path("./datasets/LD50KSplit")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
METADATA_PATH = PROJECT_ROOT / "datasets" / "landdiscover.json"
TRAIN_SPLIT_PATH = PROJECT_ROOT / "datasets" / "landdiscover50k_train.txt"
VAL_SPLIT_PATH = PROJECT_ROOT / "datasets" / "landdiscover50k_val.txt"


def _load_classes() -> list[str]:
    with METADATA_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_split_filenames(split_paths: Sequence[Path]) -> list[str]:
    filenames = []
    seen = set()
    for split_path in split_paths:
        with split_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                filename = line.strip()
                if not filename:
                    continue
                if filename in seen:
                    raise ValueError(
                        f"Duplicate LD50K filename {filename!r} found while loading {split_path}."
                    )
                seen.add(filename)
                filenames.append(filename)
    return filenames


def _build_sem_seg_records(filenames: Sequence[str]):
    image_root = DATA_ROOT / "TR_Image"
    mask_root = DATA_ROOT / "GT_ID"
    dataset_dicts = []
    for idx, filename in enumerate(filenames):
        image_path = image_root / filename
        mask_path = mask_root / filename
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing LD50K image: {image_path}")
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing LD50K mask: {mask_path}")
        dataset_dicts.append(
            {
                "file_name": str(image_path),
                "sem_seg_file_name": str(mask_path),
                "image_id": idx,
            }
        )
    return dataset_dicts


def _load_sem_seg_from_splits(split_paths: Sequence[Path]):
    return _build_sem_seg_records(_read_split_filenames(split_paths))


def _load_sem_seg_from_split(split_path: Path):
    return _load_sem_seg_from_splits((split_path,))


def register_landdiscover50k() -> None:
    classes = _load_classes()
    meta = {
        "stuff_classes": classes,
        "ignore_label": 255,
        "evaluator_type": "sem_seg",
    }
    for dataset_name, split_paths in [
        ("LandDiscover_50K_train_sem_seg", (TRAIN_SPLIT_PATH,)),
        ("LandDiscover_50K_val_sem_seg", (VAL_SPLIT_PATH,)),
        (
            "LandDiscover_50K_full_train_sem_seg",
            (TRAIN_SPLIT_PATH, VAL_SPLIT_PATH),
        ),
    ]:
        DatasetCatalog.register(
            dataset_name,
            lambda split_paths=split_paths: _load_sem_seg_from_splits(split_paths),
        )
        MetadataCatalog.get(dataset_name).set(
            image_root=str(DATA_ROOT / "TR_Image"),
            sem_seg_root=str(DATA_ROOT / "GT_ID"),
            **meta,
        )


register_landdiscover50k()
