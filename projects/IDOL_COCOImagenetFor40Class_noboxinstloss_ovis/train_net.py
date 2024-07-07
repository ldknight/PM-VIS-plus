#!/usr/bin/env python3
# Copyright (c) 2022 ByteDance. All Rights Reserved.

"""
IDOL Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import logging
import os
import sys
import itertools
import time
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results, DatasetEvaluators
from detectron2.solver.build import maybe_add_gradient_clipping

from detectron2.projects.idol import add_idol_config, build_detection_train_loader, build_detection_test_loader
from detectron2.projects.idol.data import (
    YTVISDatasetMapper, YTVISEvaluator, get_detection_dataset_dicts,DetrDatasetMapper,
  COCO_CLIP_DatasetMapper,Imagenet_CLIP_DatasetMapper
)

from detectron2.config import CfgNode
from typing import Collection, Sequence
from detectron2.utils.comm import get_world_size

logger = logging.getLogger(__name__)

import random
from collections import deque
from typing import Any, Collection, Deque, Iterable, Iterator, List, Sequence

Loader = Iterable[Any]


def _pooled_next(iterator: Iterator[Any], pool: Deque[Any]):
    if not pool:
        pool.extend(next(iterator))
    return pool.popleft()


class CombinedDataLoader:
    """
    Combines data loaders using the provided sampling ratios
    """

    BATCH_COUNT = 100

    def __init__(self, loaders: Collection[Loader], batch_size: int, ratios: Sequence[float]):
        self.loaders = loaders
        self.batch_size = batch_size
        self.ratios = ratios

    def __iter__(self) -> Iterator[List[Any]]:
        iters = [iter(loader) for loader in self.loaders]
        indices = []
        pool = [deque()] * len(iters)
        # infinite iterator, as in D2
        while True:
            if not indices:
                # just a buffer of indices, its size doesn't matter
                # as long as it's a multiple of batch_size
                k = self.batch_size * self.BATCH_COUNT
                indices = random.choices(range(len(self.loaders)), self.ratios, k=k)
            try:
                batch = [_pooled_next(iters[i], pool[i]) for i in indices[: self.batch_size]]
            except StopIteration:
                break
            indices = indices[self.batch_size :]
            yield batch

def build_combined_loader(cfg: CfgNode, loaders: Collection[Loader], ratios: Sequence[float]):
    images_per_worker = _compute_num_images_per_worker(cfg)
    return CombinedDataLoader(loaders, images_per_worker, ratios)
def _compute_num_images_per_worker(cfg: CfgNode):
    num_workers = get_world_size()
    images_per_batch = cfg.SOLVER.IMS_PER_BATCH
    assert (
        images_per_batch % num_workers == 0
    ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    assert (
        images_per_batch >= num_workers
    ), "SOLVER.IMS_PER_BATCH ({}) must be larger than the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    images_per_worker = images_per_batch // num_workers
    return images_per_worker


def filter_images_with_only_crowd_annotations(dataset_dicts, dataset_names=""):
    """
    Filter out images with none annotations or only crowd annotations
    (i.e., images without non-crowd annotations).
    A common training-time preprocessing on COCO dataset.

    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.

    Returns:
        list[dict]: the same format, but filtered.
    """
    num_before = len(dataset_dicts)

    def valid(anns):
        for ann in anns:
            if isinstance(ann, list):
                for instance in ann:
                    if instance.get("iscrowd", 0) == 0:
                        return True
            else:
                if ann.get("iscrowd", 0) == 0:
                    return True
        return False

    dataset_dicts = [x for x in dataset_dicts if valid(x["annotations"])]
    num_after = len(dataset_dicts)
    logger = logging.getLogger(__name__)
    logger.info(
        "Removed {} images with no usable annotations. {} images left.".format(
            num_before - num_after, num_after
        )
    )
    return dataset_dicts

def filter_imagenet(dataset_dict):
    ImageNet_TO_OVIS = {#19个
        1:2, 2:12, 3:13, 4:14, 5:15, 6:16, 7:17, 8:18, 9:19, 10:20, 11:12, 12:12
    }
    imagenet_flex=MetadataCatalog.get("imagenet_train_fake").thing_dataset_id_to_contiguous_id
    for frame in dataset_dict:
        annotations = frame['annotations']
        for anno_item in annotations:
            # anno_item['category_id']#序号
            imagenet_cateid=[k for k,v in imagenet_flex.items() if v==anno_item['category_id']][0]#imagenet 的分类id
            ytvis19_cateid=[v for k,v in ImageNet_TO_OVIS.items() if k==imagenet_cateid][0]
            anno_item['category_id']=ytvis19_cateid-1   #保证类别是0--39
        frame['annotations'] = annotations
    return dataset_dict
        

def filter_cocofake(dataset_dict):
    COCO_TO_OVIS = {#17个#全是真实id，即 1-person
        1:1, 2:21, 3:25, 4:22, 5:23, 6:25, 8:25, 9:24, 17:3, 18:4, 19:5, 20:6, 21:7, 22:8, 23:9, 24:10, 25:11
    }
    cocofake_cateIds=[key for key,val in COCO_TO_OVIS.items()]
    # YTVIS19_cateIds=[val for key,val in ImageNet_TO_YTVIS_2019.items()]
    coco_flex=MetadataCatalog.get("coco_train_fake").thing_dataset_id_to_contiguous_id
    # 过滤现有的coco数据集，删除不需要的数据
    index_coco = [v for k,v in coco_flex.items() if k not in cocofake_cateIds]
    for frame in dataset_dict:
        frame['annotations']=[anno_item for anno_item in frame['annotations'] if anno_item['category_id'] not in index_coco]        
    dataset_dict=[item for item in dataset_dict if len(item['annotations'])>0]
    for frame in dataset_dict:
        annotations = frame['annotations']
        for anno_item in annotations:
            # anno_item['category_id']#序号
            coco_cateid=[k for k,v in coco_flex.items() if v==anno_item['category_id']][0]#cocofake 的分类id
            ytvis19_cateid=[v for k,v in COCO_TO_OVIS.items() if k==coco_cateid][0]
            anno_item['category_id']=ytvis19_cateid-1   #保证类别是0--39
        frame['annotations'] = annotations
    return dataset_dict
        


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to YTVIS.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        elif evaluator_type == "ytvis":
            evaluator_list.append(YTVISEvaluator(dataset_name, cfg, True, output_folder))

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
        loaders = []
        mappers = []
        for d_i, dataset_name in enumerate(cfg.DATASETS.TRAIN):
            if dataset_name.startswith('coco'):
                mapper = COCO_CLIP_DatasetMapper(cfg, is_train=True)
                dataset_dict = get_detection_dataset_dicts(
                    dataset_name,
                    filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                    proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
                )
                dataset_dict=filter_cocofake(dataset_dict)
                loaders.append(build_detection_train_loader(cfg, mapper=mapper, dataset=dataset_dict))
            # elif dataset_name.startswith('ytvis'):
            #     mapper = YTVISDatasetMapper(cfg, is_train=True)

            #     dataset_dict=filter_images_with_only_crowd_annotations(dataset_dict)
            #     loaders.append(build_detection_train_loader(cfg, mapper=mapper, dataset=dataset_dict))
            #     mappers.append(mapper)
                ##################################einston
            elif dataset_name.startswith('imagenet'):
                mapper = Imagenet_CLIP_DatasetMapper(cfg, is_train=True)
                dataset_dict = get_detection_dataset_dicts(
                    dataset_name,
                    filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                    proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
                )
                dataset_dict=filter_imagenet(dataset_dict)
                loaders.append(build_detection_train_loader(cfg, mapper=mapper, dataset=dataset_dict))

        DATASET_RATIO = [1.0, 0.75]
        combined_data_loader = build_combined_loader(cfg, loaders, DATASET_RATIO)
        return combined_data_loader

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        dataset_name = cfg.DATASETS.TEST[0]
        if dataset_name.startswith('coco'):
            # mapper = CocoClipDatasetMapper(cfg, is_train=False)
            mapper = DetrDatasetMapper(cfg, is_train=False)
        elif dataset_name.startswith('ytvis'):
            mapper = YTVISDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
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


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_idol_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    # args.config_file="projects/IDOL/configs/ytvis19_r50.yaml"
    # args.num_gpus = 2
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
