第二版 IDOL-Boxinst
主要解决了 gt_inds 问题

增加config配置 boxinst中的超参数


“IDOL.py”中有所修改，使用IDOL_boxinst_inference_train替换gt中的mask(条件是box—iou足够大)


IDOL_boxinst2_inferenceMaskReplaceGtmask