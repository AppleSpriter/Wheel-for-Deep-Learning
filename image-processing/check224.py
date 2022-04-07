'''
Author: Lee Hang
检查图像尺寸是否为预定尺寸
'''

import cv2 as cv
import os

check_path = "./orig_partition/train/"
width = 224
height = 224

if __name__ == "__main__":
    grape_name_list = []
    count_num = 0
    for root, dirs, files in os.walk(check_path):
        if root == check_path:
            grape_name_list = dirs
        else:
            grape_name = grape_name_list[count_num]
            count_num += 1
            for name in files:
                path = root + "/" + name
                img = cv.imread(path)
                if img.shape[0]!=224 or img.shape[1]!=224:
                    print(path)
