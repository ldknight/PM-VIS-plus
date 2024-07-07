第二版 IDOL-Boxinst
主要解决了 gt_inds 问题

增加config配置 boxinst中的超参数


功能：IDOL-Boxinst（mask_loss+boxinst_loss）
相对于IDOL-Boxinst2 主要修改文件：idol.py（loss列表/datasets load gt_mask）  dataset_mapper.py deformable_detr.py(get_loss function)

IDOL_Boxinst2_maskloss&Boxinstloss


# 使用IDOLBoxinst2inferenceMaskReplacegtmask替换gtmask实验
# 相对于之前，主要改了：ytvis.py 文件，加载之前保存的pseudo_datasets_dict_RLE_IDOLBoxinstSwinLInferenceMask_gtBox_ytvis19

IDOL_Boxinst2_maskloss&Boxinstloss_IDOLBoxinst2inferenceMaskReplacegtmask
