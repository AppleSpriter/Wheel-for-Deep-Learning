# coding=utf-8

'''
Author: Lee Hang
Date  : 2022/02/23
对图片的背景区域以白值填充，用于分类网络输入
'''

import os
import cv2

pin_name = "14mgx/"

image_path = "../Keras-Semantic-Segmentation-master/data/pyramidclassi_224/" + pin_name
anno_path = "./dilate_out2/" + pin_name
out_path = "./exp9_out3/" + pin_name

if __name__=="__main__":
    width = 224
    height = 224
    for root, dirs, files in os.walk(anno_path):
        for file_name in files:
            #print(file_name)
            suffix = "_dilate"
            if suffix in file_name:
                the_anno_path = anno_path + file_name
                the_image_path = image_path + file_name.replace(suffix, "")

                anno_image = cv2.imread(the_anno_path, cv2.IMREAD_GRAYSCALE)
                leaf_image = cv2.imread(the_image_path)

                for i in range(height):
                    for j in range(width):
                        if (anno_image[i][j]-0)<0.01:
                            for c in range(3):
                                leaf_image[i][j][c] = 255
                name = out_path + file_name.replace(".", "_forclassi.")
                cv2.imwrite(name, leaf_image)
                print(name + " had been dealt.")
