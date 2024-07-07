第二版 IDOL-Boxinst
主要解决了 gt_inds 问题

增加config配置 boxinst中的超参数

Using segment_anything to handle the mask,and replace the IDOL_Boxinst_masks to IDOL_Boxinst_sam_masks

the mask is behand of tracking ,we repalce the target_mask by a sam_mask_res which boxes come from target_mask

update file: idol.py
add dir:     projects/IDOL/idol/segment_anything

IDOL_sam_replaceTargetMasks

    尝试在track之前替换其中质量不太好的mask
IDOL_sam_replaceTargetMasks_overMean

#为了保存包含每个segmentation的scores，准备修改ytvis-eval.py文件，增加scores字段
IDOL_addScores_ForEverySegmentation

    增加修改文件IDOL.py ytvis_eval.py