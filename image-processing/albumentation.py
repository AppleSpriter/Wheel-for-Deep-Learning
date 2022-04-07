#coding=utf-8
'''
Author: Senior

将输入地址、掩膜地址train_path、mask_path中的图像经过albumentations增强后，放在新地址
'''
import argparse
import glob
import cv2
import os
#import albumentations as A
from albumentations import (
    #PadIfNeeded,
    HorizontalFlip,    # 随机水平翻转
    VerticalFlip,      # 随机垂直翻转
    #CenterCrop,         # 中心剪裁
    RandomCrop,               # CutMix
    Compose,            # 组合？
    Transpose,          # 转置
    RandomRotate90,    # 随机90度旋转
    ElasticTransform,   # 弹性变换
    GridDistortion,     # 网络失真
    OpticalDistortion,  # 光学畸变
    RandomSizedCrop,   # 随机尺寸裁剪并缩放回原始大小
    OneOf,              # 选择性执行在它里面的变换
    CLAHE,              # 对比度首先自适应直方图均衡
    RandomBrightnessContrast,   # 随机亮度对比度
    RandomGamma,        # 随机gamma
    Cutout,              # 删除部分正方形
    ChannelShuffle      # 通道洗牌
)

target_width = 224
target_height = 224

def data_num(train_path, mask_path):
    train_img = glob.glob(train_path)
    masks = glob.glob(mask_path)
    return train_img, masks

def mask_aug():
    
    aug = Compose([
                VerticalFlip(p=0.5),
                RandomRotate90(p=0.5),
                HorizontalFlip(p=0.5),
                RandomCrop(width=int(target_width), height=int(target_height)),
                Transpose(p=0.5),
                ElasticTransform(),
                GridDistortion(),
                #OpticalDistortion(),
                CLAHE(p=0.5), # 将对比度受限的自适应直方图均衡应用于输入图像
                RandomBrightnessContrast(),
                #RandomGamma(),
                Cutout(),
                #ChannelShuffle(),
                #RandomSizedCrop(min_max_height=(128, 224), height=target_height, width=target_width, p=0.5)
                ])

    return aug

def main(whattobeal="train"):
    train_path = (r"./data/yangLingLeaves/first/" + whattobeal + "_image/*.jpg")  # 输入 img 地址
    mask_path = (r"./data/yangLingLeaves/first/" + whattobeal + "_label/*.png")  # 输入 mask 地址
    augtrain_path = (r"./data/yangLingLeaves/enhanced_first/" + whattobeal + "_image/")  # 输入增强img存放地址
    augmask_path = (r"./data/yangLingLeaves/enhanced_first/" + whattobeal + "_label/")  # 输入增强mask存放地址
    num = 20  # 输入增强图像增强的张数。
    aug = mask_aug()

    train_img, masks = data_num(train_path, mask_path)
    # 顺序修改为一样的
    train_img.sort()
    masks.sort()

    for data in range(len(train_img)):
        filename = os.path.basename(train_img[data]).replace(".jpg","")
        for i in range(num):
            image = cv2.imread(train_img[data])
            mask = cv2.imread(masks[data])
            augmented = aug(image=image, mask=mask)
            aug_image = augmented['image']
            aug_mask = augmented['mask']
            cv2.imwrite(augtrain_path + filename +  "_{}.jpg".format(i), aug_image)
            cv2.imwrite(augmask_path  + filename + "_{}.png".format(i), aug_mask)
            print("the data " + train_img[data] + " are enhanced.Count " + str(i))
# cv2.imshow("x",aug_image)
# cv2.imshow("y",aug_mask)
# cv2.waitKey(0)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    args = parser.parse_args()
    main(args.mode)
