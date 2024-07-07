# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
import io
import logging
import numpy as np
import os
from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer

from detectron2.structures import Boxes, BoxMode, PolygonMasks
from detectron2.data import DatasetCatalog, MetadataCatalog

from skimage import measure
import cv2

"""
使用SAM(gt-boxes)替换gt_masks
"""

from segment_anything import sam_model_registry, SamPredictor
sam = sam_model_registry["default"](checkpoint="/share/home/liudun/paperguides/segment-anything/sam_vit_h_4b8939.pth")
sam.to(device="cuda:0")
predictor = SamPredictor(sam)

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
    flagindex=0
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

                # if isinstance(segm, dict):
                #     if isinstance(segm["counts"], list):
                #         # convert to compressed RLE
                #         segm = mask_util.frPyObjects(segm, *segm["size"])
                # elif segm:
                #     # filter out invalid polygons (< 3 points)
                #     segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                #     if len(segm) == 0:
                #         num_instances_without_valid_segmentation += 1
                #         continue  # ignore this instance

                ############################################################################
                ## choose poly to segmentation / 4 point [x1,y1,x2,y2]      for boxes-->maskes

                ## segm = [[bbox[0],bbox[1],float(bbox[0]+bbox[2]),bbox[1],float(bbox[0]+bbox[2]),float(bbox[1]+bbox[3]),bbox[0],float(bbox[1]+bbox[3])]]
                if bbox==None:
                    num_instances_without_valid_segmentation += 1
                    continue  # ignore this instance
                
                image = cv2.imread(record['file_names'][frame_idx])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                predictor.set_image(image)    
                input_box=np.array([bbox[0],bbox[1],float(bbox[0]+bbox[2]),float(bbox[1]+bbox[3])])
                masks, _, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                #将mask转为 坐标点集
                segm=get_coco_res(masks[0].astype(int))
                # segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 4]
                if len(segm) == 0:
                    num_instances_without_valid_segmentation += 1
                    continue  # ignore this instance
                ############################################################################
                obj["segmentation"] = segm
                if id_map:
                    obj["category_id"] = id_map[obj["category_id"]]
                frame_objs.append(obj)

                flagindex+=1
                print(flagindex," ongoing...")
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
        
    save_variable(dataset_dicts,"/share/home/liudun/paperguides/VNext/projects/IDOL/dataset_dicts.txt")
    # dataset_dictsxxxxx=load_variavle("/share/home/liudun/paperguides/VNext/projects/IDOL/dataset_dicts.txt")
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
# filename = save_variable(audiodf,'audiodf.txt')
# audiodf = load_variavle('audiodf.txt')

'''
    生成mask的坐标点表示
    @param ground_truth_binary_mask np.array[true/false] eg.[255,258]  2-d
'''
def get_coco_res(ground_truth_binary_mask):
    # ground_truth_binary_mask = np.array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    #                                     [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    #                                     [  0,   0,   0,   0,   0,   1,   1,   1,   0,   0],
    #                                     [  0,   0,   0,   0,   0,   1,   1,   1,   0,   0],
    #                                     [  0,   0,   0,   0,   0,   1,   1,   1,   0,   0],
    #                                     [  0,   0,   0,   0,   0,   1,   1,   1,   0,   0],
    #                                     [  1,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    #                                     [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    #                                     [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0]], dtype=np.uint8)
    contours = measure.find_contours(ground_truth_binary_mask, 0.5)
    segment_arr=[]
    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        segment_arr.append(segmentation)
    return segment_arr

if __name__ == "__main__":
    """
    Test the YTVIS json dataset loader.
    """
    from detectron2.utils.logger import setup_logger

    logger = setup_logger(name=__name__)
    #assert sys.argv[3] in DatasetCatalog.list()
    meta = MetadataCatalog.get("ytvis_2019_train")

    json_file = "/share/home/liudun/VNext-main/datasets/ytvis_2019/annotations/instances_train_sub.json"
    image_root = "/share/home/liudun/VNext-main/datasets/ytvis_2019/train/JPEGImages"
    dicts = load_ytvis_json(json_file, image_root, dataset_name="ytvis_2019_train")
    logger.info("Done loading {} samples.".format(len(dicts)))

