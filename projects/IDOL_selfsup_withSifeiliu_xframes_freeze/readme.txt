第二版 IDOL-Boxinst
主要解决了 gt_inds 问题

增加config配置 boxinst中的超参数



based on IDOL-Boxinst2

sifeiliu selfsup.

currently only updating the file of 'segmentation_condInst.py / IDOL.py  / deformable_detr.py'

using the ref-frame&key-frame to construct the final x-frame relation

change the key&ref frame count to achieve the different x-frame which need not save the data to local

记住这个参数：SAMPLING_FRAME_NUM 2/4/6 分别表示2帧、4帧、6帧


V2.0
IDOL_selfsup_withSifeiliu_xframes_freeze
##接下来记录这个版本相对于上一个版本的改动
01--修改了文件dataset_mapper.py(200--221行)
    使得数据呈现连续性
02--修改文件 idol.py (233-234)
    只计算这一个损失：selfsup_walk
03--修改文件 train_net.py(169--191)
    冻结网络结构层
04--修改文件 idol.py(290--310)
    本来属于boxinst的，但是注释掉了，不影响

