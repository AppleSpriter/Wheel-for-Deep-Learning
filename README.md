# Wheel-for-Deep-Learning
Some image processing, data processing scripts, used to assist in-depth learning training

# Catalogue/目录

## dataset-processing

`crossfold.py` 将指定文件夹下的数据随机按照化为K折交叉，存放K份在另一文件夹下

`traintestpartition.py` 将指定文件夹下的文件按照4:1划分训练集测试集

## image-processing

`albumentation.py` 将输入路径、掩膜地址train_path、mask_path中的图像经过albumentations增强后，放在新路径

`check224.py` 检查图像尺寸是否为预定尺寸

`Cluster.py` 聚类进行特征提取分类，并根据聚类结果绘图

`getPix.py` 将分割结果图像保存在txt文件中（expired）

`HeartAroundExtraction.py` 提取图片中心区域以及边缘区域程序

`LBP.py` 根据图像通过不同的LBP局部二值模式算子生成新图

`LeafFeatureExtraction.py` 根据叶片的轮廓、面积等特征绘制柱状图

`maskimage.py` 对图片掩膜的背景区域对原图以白值填充，用于分类网络输入

`mathmorphology.py` 对图像进行数学形态学的膨胀、腐蚀操作

`resizeimage.py` 将大图进行resize（expired）

`threshold.py` OTSU自适应计算阈值，图像划分为二值

`TifToJpeg.py` 将path_source目录下的tif图片转换为jpg图片

## model-improvement

`CombineFace.py` tf中实现CosFace、ArcFace、CombineFace（Tavakoli 2021）

`Gradcam.py` 根据tf的某一层（默认最后一层）获取gradcam、gradcam++热力图

`visualpb.py` 查看pb文件

## precision-calculation

`caliou.py` 针对分割处理后的结果图和真值图计算FWIoU、iou

`calPrecision.py` 针对分割处理后的结果图和真值图计算准确率、召回率、精度、Kappa系数将图片转化为矩阵存在txt中

`GeneralMethodforClassification.py` 分类模型计算准确率、top-k情况准确率、绘制热力图、计算并绘制混淆矩阵

## windows-imagename-processing/windows图像处理

主要针对windows中图像命名统一问题，使用步骤：

1. 使用``去除空格.bat`脚本去除当前文件夹下所有文件内的空格。
2. 使用`获取文件名.bat`脚本将当前文件夹下所有文件名获取并存储到1.xls文件中。
3. 使用 `重命名.xls`将要修改的文件名进行数字自增等处理，最后使用`=D2&" "&A2&" "&B2` 其中D2为windows 修改文件名的命令`ren`，A2为原文件名，B2为需要修改的文件名。
4. 复制`重命名.xls`最后一列中的修改文件名代码到`ANSI修改.bat`中，一旦有中文情况，如果格式为UTF-8请另存为修改为ANSI运行。

