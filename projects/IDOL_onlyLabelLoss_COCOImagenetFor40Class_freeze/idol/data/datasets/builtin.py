# -*- coding: utf-8 -*-

import os
from detectron2.data.datasets.coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

from .ytvis import (
    register_ytvis_instances,
    _get_ytvis_2019_instances_meta,
    _get_ytvis_2021_instances_meta,
    _get_ovis_instances_meta,
)

from .imagenet import (
    register_imagenet_instances,
    _get_imagenet_instances_meta,
)
########################################
#einston
_PREDEFINED_SPLITS_ImageNet = {
    "imagenet_train_fake": ("imagenet/images", "imagenet/annotations/imagenet21k2ytvis2019_train.json"),
}

def register_all_imagenet(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_ImageNet.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_imagenet_instances(
            key,
            _get_imagenet_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

_PREDEFINED_SPLITS_COCOFake = {
    "coco_train_fake": ("coco/train2017", "coco/annotations/coco2ytvis2019_train.json"),
}

def register_all_cocofake(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_COCOFake.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_builtin_metadata("coco"),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

########################################

# ==== Predefined splits for YTVIS 2019 ===========
_PREDEFINED_SPLITS_YTVIS_2019 = {
    "ytvis_2019_train": ("ytvis_2019/train/JPEGImages",
                         "ytvis_2019/annotations/instances_train_sub.json"),
    "ytvis_2019_val": ("ytvis_2019/val/JPEGImages",
                       "ytvis_2019/annotations/instances_val_sub.json"),
    "ytvis_2019_test": ("ytvis_2019/test/JPEGImages",
                        "ytvis_2019/test.json"),
    "ytvis_2019_dev": ("ytvis_2019/train/JPEGImages",
                       "ytvis_2019/instances_train_sub.json"),
}


# ==== Predefined splits for YTVIS 2021 ===========
_PREDEFINED_SPLITS_YTVIS_2021 = {
    "ytvis_2021_train": ("ytvis_2021/train/JPEGImages",
                         "ytvis_2021/annotations/instances_train_sub.json"),
    "ytvis_2021_val": ("ytvis_2021/val/JPEGImages",
                       "ytvis_2021/annotations/instances_val_sub.json"),
    "ytvis_2021_test": ("ytvis_2021/test/JPEGImages",
                        "ytvis_2021/test.json"),
    "ytvis_2021_dev": ("ytvis_2021/train/JPEGImages",
                       "ytvis_2021/instances_train_sub.json"),
    "ytvis_2022_val_full": ("ytvis_2022/val/JPEGImages",
                        "ytvis_2022/instances.json"),
    "ytvis_2022_val_sub": ("ytvis_2022/val/JPEGImages",
                       "ytvis_2022/instances_sub.json"),
}


_PREDEFINED_SPLITS_OVIS = {
    "ytvis_ovis_train": ("ovis/train",
                         "ovis/annotations_train.json"),
    "ytvis_ovis_val": ("ovis/valid",
                       "ovis/annotations_valid.json"),
    "ytvis_ovis_train_sub": ("ovis/train",
                         "ovis/ovis_sub_train.json"),
    "ytvis_ovis_val_sub": ("ovis/train",
                       "ovis/ovis_sub_val.json"),
}



def register_all_ytvis_2019(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2019.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2019_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_ytvis_2021(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2021.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2021_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_ovis(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_OVIS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ovis_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )



if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_ytvis_2019(_root)
    register_all_ytvis_2021(_root)
    register_all_ovis(_root)

    register_all_imagenet(_root)
    register_all_cocofake(_root)
