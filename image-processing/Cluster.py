# coding: UTF-8
'''
Author: Lee Hang
Date  : 2022/02/23

聚类进行特征提取分类，并根据聚类结果绘图
'''
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets.samples_generator import make_blobs
from sklearn.mixture import GaussianMixture

if __name__ == "__main__":
    # 全局文件夹名称
    globalDirectory = "processingimg"
    data = []
    label = []
    # 获取文件夹下目录名称,制作data和label
    for root, dirs, files in os.walk("./datasets/labnwafu_224/" + globalDirectory + "/"):
        # 读取存储特征的npy文件
        big_tmp = np.load(globalDirectory + '.npy')
        big_data = big_tmp.tolist()
        for dir in sorted(dirs):
            counter = 0
            for theclasscount in range(len(big_data[counter])):
                # data做成[square, perimeter, shape]格式
                newList = []
                newList.append(big_data[counter][theclasscount])
                newList.append(big_data[counter + 1][theclasscount])
                newList.append(big_data[counter + 2][theclasscount])
                data.append(newList)
                label.append(str(dir))
            counter += 3

    # 设置gmm函数
    gmm = GaussianMixture(n_components=13, covariance_type='full').fit(data)
    # 训练数据
    y_pred = gmm.predict(data)
    
    for i in range(676):
        print(str(y_pred[i]) + "--true one: " + str(label[i]))

    # 绘图
    data = np.array(data)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    plt.scatter(data[:, 0], data[:, 1], c=y_pred)
    plt.show()

    

