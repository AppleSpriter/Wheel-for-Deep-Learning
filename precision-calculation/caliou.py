#coding=utf-8

'''
Author: Lee Hang
Date  : 2022/02/23

针对分割处理后的结果图和真值图计算FWIoU、iou
'''

import os
import cv2
import numpy as np

# setting profile
n_class = 2
file_name_feature = "_dilate2"
pr_path = "../Erode/output_segdata/"
gt_path = "data/yangLingLeaves/first/validation_label/"

if __name__=="__main__":
    EPS = 1e-12
    image_width = 224
    image_height = 224
    #用于测试是否代码是否和test.py 效果相同
    #pr_path = "data/output/"
    predict = []
    image_annotation = []
    # 将要进行计算的图像除以255制作50176的数组列表
    for root, dirs, files in os.walk(pr_path):
        files.sort()
        for file_name in files:
            if file_name_feature in file_name:
                if not "_ori" in file_name:
                    in_path = root + "/" + file_name
                    in_image = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
                    in_image = np.array(in_image) / 255
                    in_image = np.around(in_image)
                    predict.append(in_image)
    # 将真值图像制作数组列表
    for root, dirs, files in os.walk(gt_path):
        files.sort()
        for file_name in files:
            if not "_ori" in file_name:
                in_path = root + "/" + file_name
                in_image = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
                in_image = np.array(in_image)
                in_image = np.int32(in_image>0)
                image_annotation.append(in_image)

    tp = np.zeros(n_class)
    fp = np.zeros(n_class)
    fn = np.zeros(n_class)
    tn = np.zeros(n_class)
    n_pixels = np.zeros(n_class)

    print(len(predict))
    print(len(image_annotation))
    assert(len(predict)==len(image_annotation))

    for i in range(len(predict)):
        pr = predict[i].reshape((image_width * image_height))
        gt = image_annotation[i].reshape((image_width * image_height))
        #for i in range(224):
        #    for j in range(224):
        #        print(pr[i*224 + j])
        #print(gt)
        #break
        for c in range(n_class):
            tp[c] += np.sum((pr == c) * (gt == c))
            fp[c] += np.sum((pr == c) * (gt != c))
            fn[c] += np.sum((pr != c) * (gt == c))
            tn[c] += np.sum((pr != c) * (gt != c))
            n_pixels[c] += np.sum(gt == c)

    print(tp + fp + fn + tn)
    print(tp, fp, fn, tn)
    cl_wise_score = tp / (tp + fp + fn + EPS)
    n_pixels_norm = n_pixels / np.sum(n_pixels)
    frequency_weighted_IU = np.sum(cl_wise_score * n_pixels_norm)
    mean_IOU = np.mean(cl_wise_score)
    print("frequency_weighted_IU: ", frequency_weighted_IU)
    print("mean IOU: ", mean_IOU)
    print("class_wise_IOU:", cl_wise_score)
    print("MPA: ", sum(tp) / sum(n_pixels))

    print("准确率: ", (tp)/(tp+fn+fp))
    print("精确率: ", tp/(tp+fp))
    print("召回率: ", tp/(tp+fn))
