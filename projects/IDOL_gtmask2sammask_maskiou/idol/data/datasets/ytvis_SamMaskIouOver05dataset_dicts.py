# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
import io
import json
import logging
import numpy as np
import os
import pycocotools.mask as mask_util
from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer

from detectron2.structures import Boxes, BoxMode, PolygonMasks
from detectron2.data import DatasetCatalog, MetadataCatalog
import csv

"""
This file contains functions to parse YTVIS dataset of
COCO-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)


def load_ytvis_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    from pycocotools.ytvos import YTVOS

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        ytvis_api = YTVOS(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(ytvis_api.getCatIds())
        cats = ytvis_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    vid_ids = sorted(ytvis_api.vids.keys())
    # vids is a list of dicts, each looks something like:
    # {'license': 1,
    #  'flickr_url': ' ',
    #  'file_names': ['ff25f55852/00000.jpg', 'ff25f55852/00005.jpg', ..., 'ff25f55852/00175.jpg'],
    #  'height': 720,
    #  'width': 1280,
    #  'length': 36,
    #  'date_captured': '2019-04-11 00:55:41.903902',
    #  'id': 2232}
    vids = ytvis_api.loadVids(vid_ids)

    anns = [ytvis_api.vidToAnns[vid_id] for vid_id in vid_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(ytvis_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    vids_anns = list(zip(vids, anns))
    logger.info("Loaded {} videos in YTVIS format from {}".format(len(vids_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "category_id", "id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (vid_dict, anno_dict_list) in vids_anns:
        record = {}
        record["file_names"] = [os.path.join(image_root, vid_dict["file_names"][i]) for i in range(vid_dict["length"])]
        record["height"] = vid_dict["height"]
        record["width"] = vid_dict["width"]
        record["length"] = vid_dict["length"]
        video_id = record["video_id"] = vid_dict["id"]

        video_objs = []
        for frame_idx in range(record["length"]):
            frame_objs = []
            for anno in anno_dict_list:
                assert anno["video_id"] == video_id

                obj = {key: anno[key] for key in ann_keys if key in anno}

                _bboxes = anno.get("bboxes", None)
                _segm = anno.get("segmentations", None)

                if not (_bboxes and _segm and _bboxes[frame_idx] and _segm[frame_idx]):
                    continue

                bbox = _bboxes[frame_idx]
                segm = _segm[frame_idx]

                obj["bbox"] = bbox
                obj["bbox_mode"] = BoxMode.XYWH_ABS

                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                elif segm:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

                if id_map:
                    obj["category_id"] = id_map[obj["category_id"]]
                frame_objs.append(obj)
            video_objs.append(frame_objs)
        record["annotations"] = video_objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process. "
            "A valid polygon should be a list[float] with even length >= 6."
        )
      
    #####################################################
    # dataset_dicts=load_variavle("/share/home/liudun/VNext-main/projects/IDOL/dataset_dicts.txt")
    dataset_dicts=handle_dataset_dicts_sam(dataset_dicts)  
    # save_variable(dataset_dicts,"/share/home/liudun/VNext-main/projects/IDOL/dataset_dicts_over05_samMask.txt")
    #####################################################
    return dataset_dicts

from ast import literal_eval
def handle_dataset_dicts_sam(dataset_dicts,csv_path="/share/home/liudun/paperguides/VNext/ytvis19_gtmask2sammask_IOU_data2.csv"):
    all_count=0
    for item in dataset_dicts: #遍历每个视频
        for img_index,img_item in enumerate(item["file_names"]):    #遍历每张图片
            # item["annotations"][img_index] #标注信息
            if not len(item["annotations"][img_index]):# 标注信息为空   []
                continue
            #处理标注信息不为空
            count_img_anno=len(item["annotations"][img_index])
            count_img_anno_flag=0
            #temp_img_item=img_item.split("/share/home/liudun/VNext-main/")[1]
            with open(csv_path, encoding="utf-8-sig", mode="r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if img_item in row["filename"]:#确定目标img
                        count_img_anno_flag+=1
                        # print(float(row["IOU"])<0.8)
                        if float(row["mask_IOU"])<0.7:
                            input_box=literal_eval(row['box'])
                            # if "e+" in row["box"]:
                            #     temp_dataa=row["box"].split("[")[1].split("]")[0].split(" ")
                            #     input_box=[int(float(temp_dataa[0])),int(float(temp_dataa[1])),int(float(temp_dataa[2])),int(float(temp_dataa[3]))]
                            # else:
                            #     temp_dataa=row["box"].replace(' ','')
                            #     temp_dataa=temp_dataa.split("[")[1].split("]")[0].split(".")
                            #     input_box=[int(temp_dataa[0]),int(temp_dataa[1]),int(temp_dataa[2]),int(temp_dataa[3])]
                            target_annot_index=-1
                            for anno_index,anno_item in enumerate(item["annotations"][img_index]):
                                if int(anno_item['bbox'][0])==int(input_box[0]) and int(anno_item['bbox'][1])==int(input_box[1]) and int(anno_item['bbox'][2])==int(input_box[2]) and int(anno_item['bbox'][3])==int(input_box[3]):
                                    target_annot_index=anno_index
                            if target_annot_index!=-1:
                                # print(2222)
                                item["annotations"][img_index].pop(target_annot_index)
                                all_count+=1
                                print(all_count)
                        if count_img_anno_flag>=count_img_anno:
                            break
    return dataset_dicts

# 公共函数 变量保存至本地
import pickle
def save_variable(v,filename):
  f=open(filename,'wb')
  pickle.dump(v,f)
  f.close()
  return filename
 
def load_variavle(filename):
  f=open(filename,'rb')
  r=pickle.load(f)
  f.close()
  return r

if __name__ == "__main__":
    dataset_dicts=load_variavle("/share/home/liudun/paperguides/VNext/ytvis19_dataset_dicts_gtall_RLE.txt")
    dataset_dicts=handle_dataset_dicts_sam(dataset_dicts)  
    save_variable(dataset_dicts,"/share/home/liudun/paperguides/VNext/ytvis19_dataset_dicts_gt_samMask_maskiouOver07_RLE.txt")

    # """
    # Test the YTVIS json dataset loader.
    # """
    # from detectron2.utils.logger import setup_logger
    # from detectron2.utils.visualizer import Visualizer
    # import detectron2.data.datasets  # noqa # add pre-defined metadata
    # import sys
    # from PIL import Image

    # logger = setup_logger(name=__name__)
    # #assert sys.argv[3] in DatasetCatalog.list()
    # meta = MetadataCatalog.get("ytvis_2019_train")

    # json_file = "/share/home/liudun/VNext-main/datasets/ytvis_2019/annotations/instances_train_sub.json"
    # image_root = "/share/home/liudun/VNext-main/datasets/ytvis_2019/train/JPEGImages"
    # dicts = load_ytvis_json(json_file, image_root, dataset_name="ytvis_2019_train")
    # logger.info("Done loading {} samples.".format(len(dicts)))

    # dirname = "/share/home/liudun/VNext-main/demo/demo_res/ytvis2019/ytvis2019-train-data-vis"
    # os.makedirs(dirname, exist_ok=True)

    # def extract_frame_dic(dic, frame_idx):
    #     import copy
    #     frame_dic = copy.deepcopy(dic)
    #     annos = frame_dic.get("annotations", None)
    #     if annos:
    #         frame_dic["annotations"] = annos[frame_idx]

    #     return frame_dic

    # for d in dicts:
    #     vid_name = d["file_names"][0].split('/')[-2]
    #     os.makedirs(os.path.join(dirname, vid_name), exist_ok=True)
    #     for idx, file_name in enumerate(d["file_names"]):
    #         img = np.array(Image.open(file_name))
    #         visualizer = Visualizer(img, metadata=meta)
    #         vis = visualizer.draw_dataset_dict(extract_frame_dic(d, idx))
    #         fpath = os.path.join(dirname, vid_name, file_name.split('/')[-1])
    #         vis.save(fpath)
