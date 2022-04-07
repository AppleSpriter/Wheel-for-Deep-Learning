'''
Author: Lee Hang

对图像进行数学形态学的膨胀、腐蚀操作
'''

import cv2 as cv
import os

def grayOperation(image):
    return cv.cvtColor(image, cv.COLOR_RGB2GRAY), "gray"

def binaryOperation(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    return binary, "binary"

def dilateOperation(image, iteration=1):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.dilate(binary, kernel=kernel, iterations=iteration)
    return dst, "dilate"

def erodeOperation(image, iteration=1):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.erode(binary, kernel=kernel, iterations=iteration)
    return dst, "erode"

def openOperation(image, iteration=1):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.erode(binary, kernel=kernel, iterations=iteration)
    dst = cv.dilate(dst, kernel=kernel, iterations=iteration)
    return dst, "open"

def closeOperation(image, iteration=1):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.dilate(binary, kernel=kernel, iterations=iteration)
    dst = cv.erode(dst, kernel=kernel, iterations=iteration)
    return dst, "close"

if __name__ == "__main__":
    prefix = "14mgx/"
    output_path_source = "./dilate_out2/" + prefix
    file_path = "../Keras-Semantic-Segmentation-master/data/dsunet_out1/" + prefix
    #file_path = "./crossfold5/"
    iteration = 1
    grape_name_list = []
    #count_num = 0
    for root, dirs, files in os.walk(file_path):
        #grape_name = grape_name_list[count_num]
        #count_num += 1
        for file_name in files:
            if not "_ori" in file_name:
                in_path = root + "/" + file_name
                if in_path.endswith(".jpg") or in_path.endswith(".png"):
                    in_image = cv.imread(in_path)
                    #dealed_image, suffix = binaryOperation(in_image)
                    #dealed_image, suffix = grayOperation(in_image)
                    #dealed_image, suffix = erodeOperation(in_image)
                    dealed_image, suffix = dilateOperation(in_image)
                    #dealed_image, suffix = openOperation(in_image, iteration)
                    #dealed_image, suffix = closeOperation(in_image, iteration)
                    output_path = output_path_source + file_name.replace(".jpg", "_") + suffix + ".jpg"
                    # 分离文件名和路径
                    fpath, fname = os.path.split(output_path)
                    if not os.path.exists(fpath):
                        os.makedirs(fpath)
                    cv.imwrite(output_path, dealed_image)
                    print(output_path + " is proceeded.")
        break
