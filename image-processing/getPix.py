# -*- coding: utf-8-unix -*-
'''
Author: Lee Hang
Date  : 2019/03/01

将分割结果图像保存在txt文件中
'''

from PIL import Image
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2 

defaultPath = "pred_5.png"

def getPngPix(pngPath = defaultPath,pixelX = 1,pixelY = 1):
    img_src = Image.open(pngPath)
    img_src = img_src.convert('RGBA')
    img = cv2.imread(pngPath)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # str_strlist = img_src.load()
    str_strlist = gray
    data = str_strlist[pixelY,pixelX]
    img_src.close()
    return data

def imgToMatrix():
    f = open("imgInfo.txt", "w")
    img = np.array(Image.open(defaultPath))
    
    # 查看图片
    plt.show()

    # 输入图片参数
    print img.shape
    print img.dtype 
    print img.size 
    print type(img)
    # asimg = img.around()
    # np.set_printoptions(precision=0)
    
    # print asimg.dtype

    # 保存图片
    np.savetxt("npImgInfo.txt", img, fmt="%.0f")
    '''
    for i in range(1,224):
        for j in range(1,224):
            # print(img[i,j]),
            f.write(img[i,j])
        f.write("\n") 
    '''

    f.close()

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
    if '-i' in mydic.keys():
        defaultPath = mydic['-i']
    
    print("Image is " + defaultPath) 

    # imgToMatrix()
    tmp1 = [[0, 0, 0],
            [255, 255, 0]
            [0, 255, 255]]

    tmp2 = [[255, 0, 255],
            [0, 255, 255],
            [0, 255, 255]]

    tmp3 = tmp1 + tmp2
    print(tmp3)


'''
for pixelX in range(1,224):
    for pixelY in range(1,224):
        print(getPngPix(), pixelX, pixelY)

img = cv2.imread("pred_5.png")
print(img)
'''
