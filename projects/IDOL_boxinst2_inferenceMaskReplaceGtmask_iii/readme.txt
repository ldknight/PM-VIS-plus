第二版 IDOL-Boxinst
主要解决了 gt_inds 问题

增加config配置 boxinst中的超参数


“IDOL.py”中有所修改，使用IDOL_boxinst_inference_train替换gt中的mask(条件是box—iou足够大)

# IDOL_boxinst2_inferenceMaskReplaceGtmask



相对于上一代依然更新'idol.py'，将进一步细化规范获得更加准确的inference_mask
IDOL_boxinst2_inferenceMaskReplaceGtmask_ii

*** 这里面的‘idol.py’==>instance_index!=instance_index2 不合理，需要改一下
*** 里面没有重复instance删除机制，删除模式参考run_VNext.ipynb

# 生成第三代数据集
IDOL_boxinst2_inferenceMaskReplaceGtmask_iii

