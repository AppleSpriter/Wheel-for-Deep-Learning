# -*- coding: utf-8 -*-

'''
Author: Lee Hang
Date  : 2020/02/23

OTSU自适应计算阈值，图像划分为二值
'''

import cv2
import sys
from PIL import Image

thresh_num = 0

def getargvdic(argv):
    optd = {}
    while argv:
        if argv[0][0] == '-':
            optd[argv[0]] = argv[1]
            argv = argv[2:]
        else:
            argv = argv[1:]
    return optd


if __name__ == '__main__':
    argv = sys.argv
    mydic = getargvdic(argv)

    # 读取图片
    if '-image' in mydic.keys():
        inPath = mydic['-image']
    print("The input image is " + inPath) 

    # 输入保存图片
    if '-output' in mydic.keys():
        outPath = mydic['-output']
    print("The outPut image is " + outPath)

    img = cv2.imread(inPath, 0)
    # img = cv2.imread('groundTruth.png', 0)
    ret, thresh1 = cv2.threshold(img, thresh_num, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img, thresh_num, 255, cv2.THRESH_BINARY_INV)  # （黑白二值反转）
    # thresh3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 0)

    # print(ret)
    # cv2.imshow('threshold image', thresh1)
    '''
    cv2.imshow('thresh2', thresh2)
    cv2.imshow('grey-map', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    cv2.imwrite(outPath, thresh1)
    # cv2.imwrite("thresh2.png", thresh2)
