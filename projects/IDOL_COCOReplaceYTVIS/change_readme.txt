IDOL 原始文件
IDOL_origin

IDOL_COCOReplaceYTVIS
V1.0
使用coco训练模型，使得模型具备40类的识别能力；
1- 使用的数据集：【coco_2017_train_fake】（不清楚数据加载的时候，能不能自动分出关键帧和参考帧，也就是同一张图片的两种状态）
2- 使用的损失函数：原始IDOL的相关损失函数；
3- 考虑加入BoxInst的相关损失函数，试试能否提高分割能力；
4- 训练YTVIS_train,使用sifeiliuloss，考虑清楚是否冻结其他层；