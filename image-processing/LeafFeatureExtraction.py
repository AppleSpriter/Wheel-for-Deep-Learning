# coding: UTF-8
'''
Author: Lee Hang
Date  : 2020/08/05

根据叶片的轮廓、面积等特征绘制柱状图
'''
import cv2
import os
import sys
import time 
import numpy as np
import logging 
import matplotlib.pyplot as plt
from alive_progress import alive_bar

# 图像长宽
IMAGE_SHAPE = 224

#扫描输出图片列表
def each_file(filepath):
    list = []
    pathDir = os.listdir(filepath)
    pathDir.sort()
    for allDir in pathDir:
        child = os.path.join(filepath, allDir)
        list.append(child)
    return list

# 画出柱状图
def draw_bar(list, variety, sorp):
    # 定义柱状图间隔
    if sorp == "square":
        interval = 2000
    elif sorp == "perimeter":
        interval = 50
    elif sorp == "shape":
        interval = 15
    else:
        print("please make sorp correct")
        return 
    drawData = np.zeros(7)
    mean = np.mean(list)
    # 计算画图数据的值
    for i in list:
        if i < (mean - 2.5 * interval):
            drawData[0] += 1
        elif i < (mean - 1.5 * interval):
            drawData[1] += 1
        elif i < (mean - 0.5 * interval):
            drawData[2] += 1
        elif i < (mean + 0.5 * interval):
            drawData[3] += 1
        elif i < (mean + 1.5 * interval):
            drawData[4] += 1
        elif i < (mean + 2.5 * interval):
            drawData[5] += 1
        else:
            drawData[6] += 1
    # 绘图
    name_list = np.array(["less than " + str(int(mean - 2.5 * interval)), \
            str(int(mean - 2.5 * interval)) + "-" + str(int(mean - 1.5 * interval)),\
            str(int(mean - 1.5 * interval)) + "-" + str(int(mean - 0.5 * interval)), \
            str(int(mean - 0.5 * interval)) + "-" + str(int(mean + 0.5 * interval)), \
            str(int(mean + 0.5 * interval)) + "-" + str(int(mean + 1.5 * interval)), \
            str(int(mean + 1.5 * interval)) + "-" + str(int(mean + 2.5 * interval)), \
            "more than " + str(int(mean + 2.5 * interval))])
    plt.rcParams['figure.figsize'] = (16.0, 8.0)
    plt.title(variety + " " + sorp)
    plt.bar(range(len(drawData)), drawData, tick_label=name_list)
    plt.savefig("figures/test/" + variety + "-" + sorp + ".jpg")
    # plt.show()

# 获取最长且非四周的外围边缘
def edge_extraction(list_contours):
    max_line = []
    sec_line = []
    # 找到最长和第二长的两个array list
    for i in list_contours:
        if len(i) > len(max_line):
            max_line = i
        if len(i) > len(sec_line) and len(i) != len(max_line):
            sec_line = i
    """
    # 本来想去掉最外层的一圈,但是好像是轮廓加上四个角
    if [0, 0] in max_line and [IMAGE_SHAPE - 1, IMAGE_SHAPE - 1] in max_line:
        return sec_line
    else:
        return max_line
    """
    return max_line

# 叶片轮廓提取
def perimeter_calculate(img):
    # 获取灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化
    ret, binary = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
    # 展示2值图像
    # cv2.imshow("origin", binary)
    # 轮廓提取
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, \
            cv2.CHAIN_APPROX_SIMPLE)
    """
    # 轮廓红色标记,查看标记后图片
    cv2.drawContours(img, edge_extraction(contours), -1, (0, 0, 255), 3)
    print(edge_extraction(contours))
    print("image perimeter: %d" %(len(edge_extraction(contours)) - 12))
    cv2.imshow("img", img)
    cv2.waitKey(0)
    """
    return (len(edge_extraction(contours)) - 12)

# 叶片面积计算
def square_calculate(img):
    # 叶片面积
    square = 0
    # 计算不为白色的叶片面积
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if all(img[i][j] != [255, 255, 255]):
                square += 1

    return square

if __name__ == "__main__":
    # 全局文件夹名称
    globalDirectory = "processingimg"
    # 获取文件夹下所有目录名称
    for root, dirs, files in os.walk("./datasets/labnwafu_224/" + globalDirectory +  "/"): 
        if sys.argv[1] == "read":
            # 读取存储的npy文件
            big_tmp = np.load(globalDirectory + '.npy')
            big_data = big_tmp.tolist()
            counter = 0
            for dir in sorted(dirs):
                grape_name = str(dir)
                # 计算该类叶片的平均面积和周长
                square_list = big_data[counter]
                counter += 1
                perimeter_list = big_data[counter]
                counter += 1
                shape_list = big_data[counter]
                counter += 1
                # 输出平均长度和面积
                print("----------------------------------------------------------")
                print(grape_name + " square: " + str(np.mean(square_list)))
                print(grape_name + " perimeter: " + str(np.mean(perimeter_list)))
                print(grape_name + " shape: " + str(np.mean(shape_list)))
                # 画图
                # draw_bar(square_list, grape_name, "square")
                # draw_bar(perimeter_list, grape_name, "perimeter")
                # draw_bar(shape_list, grape_name, "shape")
            # 跳出walk循坏
            break

        elif sys.argv[1] == "execute":
            big_data = []
            for dir in sorted(dirs):
                grape_name = str(dir)
                # 定义输入输出文件路径
                input_path0 = r"./datasets/labnwafu_224/" + globalDirectory + "/" + grape_name +  "/"
                # 将输入文件路径转化为图片列表
                input_path = each_file(input_path0)
                # 计算该类叶片的平均面积和周长
                square_list = []
                perimeter_list = []
                shape_list = []
                with alive_bar(len(input_path)) as bar:
                    for input in input_path:
                        image = cv2.imread(input)
                        square_list.append(square_calculate(image))
                        perimeter_list.append(perimeter_calculate(image))
                        shape_list.append(square_calculate(image) / perimeter_calculate(image))
                        bar()
                # 输出平均长度和面积
                print("----------------------------------------------------------")
                print(grape_name + " square: " + str(np.mean(square_list)))
                print(grape_name + " perimeter: " + str(np.mean(perimeter_list)))
                print(grape_name + " shape: " + str(np.mean(shape_list)))
                # 画图
                # draw_bar(square_list, grape_name, "square")
                # draw_bar(perimeter_list, grape_name, "perimeter")
                # draw_bar(shape_list, grape_name, "shape")
                # 数据写入big_data列表
                big_data.append(square_list)
                big_data.append(perimeter_list)
                big_data.append(shape_list)

            # 储存列表为npy文件
            big_tmp = np.array(big_data)
            np.save(globalDirectory + '.npy', big_tmp)
            # 跳出walk循坏
            break
        
        else:
            print("请输入正确参数指令")
            # 跳出walk循坏
            break
