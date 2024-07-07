第二版 IDOL-Boxinst
主要解决了 gt_inds 问题

增加config配置 boxinst中的超参数

Using segment_anything to handle the mask,and replace the IDOL_Boxinst_masks to IDOL_Boxinst_sam_masks

the mask is before the tracking ,we repalce the before_mask by a sam_mask_res which boxes come from IDOL_pred_boxes

update file: idol.py
add dir:     projects/IDOL/idol/segment_anything