第二版 IDOL-Boxinst
主要解决了 gt_inds 问题

增加config配置 boxinst中的超参数


IDOL_boxinst2

# IDOL_boxinst2&maskloss_COCOImagenetFor40Class
    在coco上使用maskloss和boxinstloss
    在imagenet上使用boxinstloss

# 不使用boxisntloss

IDOL_COCOImagenetFor40Class_noboxinstloss_ytvis21

修改文件：
    1-train_net.py（ImageNet_TO_YTVIS_2021、COCO_TO_YTVIS_2021）
    2-imagenet.py （imagenet_CATEGORIES）
    3-builtin.py  （_PREDEFINED_SPLITS_ImageNet 、 _PREDEFINED_SPLITS_COCOFake ）
    4-idol.py (公共注释，注释掉了276～296行)

    