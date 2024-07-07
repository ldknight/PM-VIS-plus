第二版 IDOL-Boxinst
主要解决了 gt_inds 问题

增加config配置 boxinst中的超参数


功能：IDOL-Boxinst（mask_loss+boxinst_loss）
相对于IDOL-Boxinst2 主要修改文件：idol.py（loss列表/datasets load gt_mask）  dataset_mapper.py deformable_detr.py(get_loss function)

ytvis.py文件有改动 loadData注意

IDOL_Boxinst2_maskloss&Boxinstloss_loadDatasetsDict

"loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes)*0.5,
                "loss_dice": dice_loss(src_masks, target_masks, num_boxes)*0.5,

                deformable_detr.py 有改动 注意⚠️

################
将load data方式由txt改为json加载，相对于之前主要修改ytvis.py\builtin.py

使用时候只需要修改builtin.py文件（有两处需要修改）

PMVIS_loadJsonData
