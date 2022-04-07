# -*- coding: utf-8-unix -*-

'''
Author: Lee Hang
Date  : 2019/02/23

将大图进行resize
'''

from __future__ import division

from PIL import Image
import sys
import os
import numpy as np
import cv2

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

    if '-image' in mydic.keys():
        imgPath = mydic['-image']

    img = cv2.imread(imgPath, -1)

    height, width = img.shape[:2]

    # 缩小图像
    size = (int(width*0.224), int(height*0.224))
    shrink = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    cv2.imshow("src", img)
    cv2.imshow("shrink", shrink)

    cv2.waitKey(0)

    cv2.imwrite("shrink.png", shrink)

