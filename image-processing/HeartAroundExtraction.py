# coding: UTF-8
'''
Author: Lee Hang
Date  : 2021/03/23

提取叶片中心区域以及边缘区域程序
'''
import cv2
import os
import time
import numpy as np
import logging
from alive_progress import alive_bar

createSize = 224

def each_file(filepath):
    '''
    扫描输出图片列表
    :param filepath: 输入文件夹路径
    '''
    list = []
    pathDir = os.listdir(filepath)
    pathDir.sort()
    logging.debug("pathDir: %s" %(pathDir))
    for allDir in pathDir:
        child = os.path.join(filepath, allDir)
        list.append(child)
    return list

def resize(image):
    '''
    图片重新划分大小,用于划分为224尺寸方便网络训练
    :param image: 用于更改尺寸的图片
    '''
    size = createSize
    return cv2.resize(image, (size, size))

def rename(prefix, suffix, name_count):
    '''
    统一长度的重命名
    :param prefix: 文件名前缀
    :param suffix: 文件名后缀,主要用于区分原始图、边缘、叶心
    :param name_count: 命名顺序
    '''
    if suffix != "":
        suffix = "_" + suffix #判断后缀不为空才加上

    if name_count < 10:
        name = prefix + "-00" + str(name_count) + suffix + ".jpg"
    elif name_count < 100:
        name = prefix + "-0" + str(name_count) + suffix + ".jpg"
    else:
        name = prefix + "-" + str(name_count) + suffix + ".jpg"
    return name
        
def cv2_crop(im, box):
    '''
    cv2实现类似PIL的裁剪
    :param im: 加载好的图像
    :param box: 裁剪的矩形，元组(left, upper, right, lower).
    '''
    return im.copy()[box[1]:box[3], box[0]:box[2], :]

def HeartAroundExtraction(input_path, output_path):
    '''
    分离叶片中心以及边缘,并制作为模型训练数据集
    其中边缘叶片区域用白色覆盖
    :param input_path: 图片的输入路径
    :param output_path: 处理完成图像输出路径
    '''
    name_count = 1
    with alive_bar(len(input_path), spinner="balls_scrolling") as bar:
        for input in input_path:
            img = cv2.imread(input)
            #矩形切割区域
            #img_quarter = int(img.shape[0]/4)       #获取图片1/4边长
            #box = (img_quarter, img_quarter, 3 * img_quarter, 3 * img_quarter)  #切分盒,中间切1/4-3/4正方形
            #around_region = cv2.rectangle(img.copy(), (box[0], box[1]), (box[2], box[3]),
            #        (0, 0, 0), -1)            #提取边缘区域
            #around_region = resize(around_region)
            #heart_region = resize(cv2_crop(img, box))   #提取叶心中间区域,并将其变成全图大小
            #origin = resize(img)
            #创建纯黑rgb图
            #b_channel = np.ones((createSize, createSize), dtype=np.uint8) * 0
            #g_channel = np.ones((createSize, createSize), dtype=np.uint8) * 0
            #r_channel = np.ones((createSize, createSize), dtype=np.uint8) * 0
            #blackImage = cv2.merge((b_channel, g_channel, r_channel))
            #blackImage = cv2.rectangle(blackImage, (box[0], box[1]), (box[2], box[3]),
            #        (255, 255, 255), -1)    #纯黑中间抠出box的白色
            #heart_region = resize(img)
            #heart_region = cv2.bitwise_and(blackImage, img.copy())  #按位与运算,提取中心区域,周围为黑色
            #圆形切割区域
            center = int(img.shape[0] / 2)  #获取圆心(图片中心)
            radius = int(img.shape[0] / 5)  #计算半径
            around_region = cv2.circle(img.copy(), (center, center), radius, (0,0,0), -1)
            #输出正在处理的图像
            print("The " + grape_name + " image " + str(name_count) + " finished")
            #cv2.imwrite(rename(output_path + grape_name, "origin", name_count), origin)
            cv2.imwrite(rename(output_path + grape_name, "around_region", name_count), around_region)
            #cv2.imwrite(rename(output_path + grape_name, "heart_region", name_count), heart_region)
            # 用户名定义
            name_count += 1
            bar()

def HeartAroundExtractionOneImage(input_path, output_path):
    '''
    分离单张叶片中心以及边缘,并制作为模型训练数据集
    其中边缘叶片区域用白色覆盖
    :param input_path: 图片的输入路径
    :param output_path: 处理完成图像输出路径
    '''
    img = cv2.imread(input_path)
    #矩形切割区域
    #img_quarter = int(img.shape[0]/4)       #获取图片1/4边长
    #box = (img_quarter, img_quarter, 3 * img_quarter, 3 * img_quarter)  #切分盒,中间切1/4-3/4正方形
    #around_region = cv2.rectangle(img.copy(), (box[0], box[1]), (box[2], box[3]),
    #        (0, 0, 0), -1)            #提取边缘区域
    #around_region = resize(around_region)
    #heart_region = resize(cv2_crop(img, box))   #提取叶心中间区域,并将其变成全图大小
    #origin = resize(img)
    #创建纯黑rgb图
    #b_channel = np.ones((createSize, createSize), dtype=np.uint8) * 0
    #g_channel = np.ones((createSize, createSize), dtype=np.uint8) * 0
    #r_channel = np.ones((createSize, createSize), dtype=np.uint8) * 0
    #blackImage = cv2.merge((b_channel, g_channel, r_channel))
    #blackImage = cv2.rectangle(blackImage, (box[0], box[1]), (box[2], box[3]),
    #        (255, 255, 255), -1)    #纯黑中间抠出box的白色
    #heart_region = resize(img)
    #heart_region = cv2.bitwise_and(blackImage, img.copy())  #按位与运算,提取中心区域,周围为黑色
    #圆形切割区域
    center = int(img.shape[0] / 2)
    radius = int(img.shape[0] / 6)
    around_region = cv2.circle(img.copy(), (center, center), radius, (0,0,0), -1)
    name_count = 1
    # 输出正在处理的图像
    print("The " + input_path + " image " + str(name_count) + " finished")
    cv2.imshow("test image", around_region)
    cv2.waitKey(0)
    #cv2.imwrite(rename(output_path + grape_name, "origin", name_count), origin)
    #cv2.imwrite(rename(output_path + grape_name, "around_region", name_count), around_region)
    #cv2.imwrite(rename(output_path, "heart_region", name_count), heart_region)
    # 用户名定义

def ChangeLeavesName(input_path, output_path):
    '''
    读取图片并重命名
    :param input_path: 图片的输入路径
    :param output_path: 处理完成图像输出路径
    '''
    name_count = 1
    with alive_bar(len(input_path), spinner="balls_scrolling") as bar:
        for input in input_path:
            img = cv2.imread(input)
            #输出正在处理的图像
            print("The " + grape_name + " image " + str(name_count) + " finished")
            cv2.imwrite(rename(output_path + grape_name, "", name_count), img)
            # 用户名定义
            name_count += 1
            bar()



if __name__ == "__main__":
    # 品种名
    global grape_name
    # 日志设置
    logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
    logging.debug("程序测试")
    logging.disable(logging.CRITICAL)
    test_one_path = r"./datasets/labnwafu_224/train/14wk/14lwk091.jpg"
    test_one_output = "./datasets/around_black_circle_224"
    #HeartAroundExtractionOneImage(test_one_path, test_one_output)
    # 获取文件夹下所有目录名称
    target_folder = "./datasets/labnwafu_224_tmp/test/"
    for root, dirs, files in os.walk(target_folder): 
        for dir in dirs:
            grape_name = str(dir)
            # 定义输入输出文件路径
            input_path0 = target_folder + grape_name +  "/"
            #output_path = r"./datasets/around_test_black_circle_224/" + grape_name + "/"
            changename_path = r"./datasets/labnwafu_224_LANCZOS5/test/" + grape_name + "/"
            input_path = each_file(input_path0)     # 将输入文件路径转化为图片列表
            #HeartAroundExtraction(input_path, output_path)  # 分离叶片中心以及边缘数据集
            ChangeLeavesName(input_path, changename_path)
