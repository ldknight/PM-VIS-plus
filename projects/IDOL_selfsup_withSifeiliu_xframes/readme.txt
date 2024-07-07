第二版 IDOL-Boxinst
主要解决了 gt_inds 问题

增加config配置 boxinst中的超参数



based on IDOL-Boxinst2

# IDOL_selfsup_withSifeiliu_xframes

sifeiliu selfsup.

currently only updating the file of 'segmentation_condInst.py / IDOL.py  / deformable_detr.py'

using the ref-frame&key-frame to construct the final x-frame relation

change the key&ref frame count to achieve the different x-frame which need not save the data to local

这个版本里的frame都是不连续的，不知道是不是因为这个所以准确率低
