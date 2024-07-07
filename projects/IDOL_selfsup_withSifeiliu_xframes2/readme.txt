第二版 IDOL-Boxinst
主要解决了 gt_inds 问题

增加config配置 boxinst中的超参数



based on IDOL-Boxinst2

sifeiliu selfsup.

currently only updating the file of 'segmentation_condInst.py / IDOL.py  / deformable_detr.py'

using the ref-frame&key-frame to construct the final x-frame relation

change the key&ref frame count to achieve the different x-frame which need not save the data to local

IDOL_selfsup_withSifeiliu_xframes

#自监督v2.0 
IDOL_selfsup_withSifeiliu_xframes2

1-去掉新加的用于自监督的头，全部来源于mask—head
2-修改了文件dataset_mapper.py(200--221行)
    使得数据呈现连续性