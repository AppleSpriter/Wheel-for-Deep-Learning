# -*- coding: utf-8-unix -*-
'''
Author: Lee Hang
Date  : 2018/12/11

针对分割处理后的结果图和真值图计算准确率、召回率、精度、Kappa系数
将图片转化为矩阵存在txt中

'''
from __future__ import division

from PIL import Image
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2 

testPath = "test.png"
truePath = "true.png"


def imgToMatrix(trueIm, testIm):
    f1 = open("npImgInfo_true.txt", "w")
    trueImg = np.array(Image.open(trueIm))
    
    '''
    # 查看图片信息
    plt.show()

    # 输入图片参数
    print img.shape
    print img.dtype 
    print img.size 
    print type(img)
    '''

    # 保存图片
    np.savetxt("npImgInfo_true.txt", trueImg, fmt="%.0f")
    f1.close()

    testImg = np.array(Image.open(testIm))
    np.savetxt("npImgInfo_test.txt", testImg, fmt="%.0f")

    calPre(trueImg, testImg)


def getargvdic(argv):
    optd = {}
    while argv:
        if argv[0][0] == '-':
            optd[argv[0]] = argv[1]
            argv = argv[2:]
        else:
            argv = argv[1:]
    return optd


def calPre(trueImg, testImg):
    '''
    插值计算
    0 双方都判断不是梯田
    255 误判成梯田区域
    254 没有识别出的梯田区域
    253 判断梯田识别准确地区域
    '''
    Dvalue = trueImg * 2 + testImg 
    np.savetxt("npImgInfo_dvalue.txt", Dvalue, fmt="%.0f")

    # 统计区域像素点
    sumPoint = 224 * 224
    rs0 = np.sum(Dvalue==0)
    rs255 = np.sum(Dvalue==255)
    rs254 = np.sum(Dvalue==254)
    rs253 = np.sum(Dvalue==253)

    # 输出参数个数
    print("总共有像素点: ")
    print(sumPoint)
    print("分类结果为非梯田,真值为非梯田的像素点有: " )
    print(rs0)
    print("分类结果是梯田,真值为非梯田的像素点有: " )
    print(rs255)
    print("分类结果是非梯田,真值为梯田像素点有: " )
    print(rs254)
    print("分类结果为梯田,真值为梯田的像素点有: " )
    print(rs253) 
    print("\n")

    # 输出准确率
    accuracyRate = (rs253 + rs0) / sumPoint
    precision = (rs253) / (rs253 + rs255) 
    recallRate = (rs253) / (rs253 + rs254)
    tmpPe = ((rs253 + rs254) * (rs253 + rs255) + (rs255 + rs0) * (rs254 + rs0)) / (sumPoint * sumPoint)
    Kappa = (accuracyRate - tmpPe) / (1 - tmpPe)

    print("正确率为: ")
    print(accuracyRate)
    print("召回率为: ")
    print(recallRate)
    print("Kappa系数为: ")
    print(Kappa)
    print("精度为：")
    print(precision)

    print("Pe为: ")
    print(tmpPe)



if __name__ == '__main__':
    argv = sys.argv
    mydic = getargvdic(argv)

    # 读取真值图片
    if '-true' in mydic.keys():
        truePath = mydic['-true']
    print("The true value image is " + truePath) 

    # 读取测试图片
    if '-test' in mydic.keys():
        testPath = mydic['-test']
    print("The test value image is " + testPath)

    imgToMatrix(truePath, testPath)
    
    '''
    # 学习测试
    tmp1 = [[0, 0, 0],
            [0, 255, 255],
            [255, 255, 0]]

    tmp2 = [[255, 0, 0],
            [0, 255, 0],
            [255, 0, 255]]

    tmp3 = tmp1 + tmp2 
    print(tmp3)
    ''' 
    


