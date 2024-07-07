# 该文件主要通过在inference时进行可视化（该文件为IDOL原始文件）
修改文件 idol.py(在ytvis inference中)
     config.py(增加 cfg.MODEL.IDOL.save_visual_res_path )

通过以下推理阶段 即可实现可视化
python3 /share/home/liudun/paperguides/VNext/projects/IDOL/train_net.py --config-file /share/home/liudun/paperguides/VNext/outdir/IDOL_YTVIS19_R50_boxinst2/config.yaml --num-gpus 4 --eval-only --dist-url tcp://127.0.0.1:50155 MODEL.WEIGHTS /share/home/liudun/paperguides/VNext/outdir/IDOL_YTVIS19_R50_boxinst2/model_0044999.pth MODEL.IDOL.save_visual_res_path "/share/home/liudun/paperguides/VNext/demo_outdir/IDOL_Boxinst2_r50_ytvis19/"


# IDOL_visiableytvis_images 推理可视化文件