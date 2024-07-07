第二版 IDOL-Boxinst
主要解决了 gt_inds 问题

增加config配置 boxinst中的超参数


功能：IDOL-Boxinst（mask_loss+boxinst_loss）
相对于IDOL-Boxinst2 主要修改文件：idol.py（loss列表/datasets load gt_mask）  dataset_mapper.py deformable_detr.py(get_loss function)

使用SAM生成的mask，加上boxinst2的box监督，训练IDOL网络，获得box监督条件下在ytvis19上的准确率。
其中，损失函数除了全监督条件下的maskloss，还增加了boxinst的两个损失函数pairwiseloss、predictloss
其中，dataset_dicts_over05_samMask_RLE等皆为sam生成的mask替换了gt_mask，并且mask转化为RLE格式。

dataset_mapper.py \ idol.py \ ytvis.py \ deformable_detr.py

IDOL_Boxinst2_maskloss&Boxinstloss_samIOU_RLE_divideIOU

loss_mask / iou