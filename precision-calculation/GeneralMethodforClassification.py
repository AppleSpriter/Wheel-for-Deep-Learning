# coding: UTF-8
'''
Author: Lee Hang
Date  : 2021/03/04

分类模型计算准确率、top-k情况准确率、绘制热力图、计算并绘制混淆矩阵
'''

import itertools
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt

# 计算准确率
def acc(list1, list2):
    if len(list1)!=len(list2):
            print("acc函数输入长度不等!")
            return     

    sameElements = 0
    for i in range(len(list1)):
            if int(list1[i]) == int(list2[i]):
                    sameElements += 1

    print("acc为: %f" % (sameElements/len(list1)))
    return (sameElements/len(list1))


# 计算k情况准确率
def acc_k(list1, list2, k, pr=[], re=[]):
    if len(list1)!=len(list2):
        print("acc函数输入长度不等!")
        return     
    sameElements = 0
    if len(pr) != 0:
        precision = sum(pr)/len(pr)
        recall = sum(re)/len(re)
        if (precision+recall) == 0:
            f1score = -1
        else:
            f1score = 2*precision*recall/(precision+recall)
        print("总体Precision为：%f" % precision)
        print("总体Recall为：%f" % recall)
        print("总体F1-score为：%f" % f1score)

    for i in range(len(list1)):
        for j in range(k):
            if int(list1[i][j]) == int(list2[i]):
                sameElements += 1
    print("\n总体acc为: %f" % (sameElements/len(list1)))

    return (sameElements/len(list1))


# 计算不同类的准确率、精度和召回率,找到最容易出错的一个点,仅支持ak==1
def pre_rec_for_each(list1, list2, test_images, name_dic):
    if len(list1)!=len(list2):
        print("计算精度召回率函数输入长度不等!")
        return
    # 将tensor转化为list
    list3 = []
    for i in range(len(list2)):
        list3.append(int(list2[i]))
    paper_one_dict_rev = {5:0, 8:1, 7:2, 9:3, 0:4, 10:5, 1:6, 2:7, 3:8, 4:9, 6:10}
    #创建列表储存所有类的指标
    pr_list = [0] * len(set(list3))
    re_list = [0] * len(set(list3))

    for i in set(list3):
        #按照论文顺序输出
        #i = paper_one_dict_rev[i]
        # 初始化中间参数
        sameElements = 0
        testElements = 0
        trueElements = 0
 
        for j in range(len(list1)):
            # 统计测试和真值一致的某类数量
            if int(list1[j][0]) == int(list2[j]) and int(list2[j]) == i:
                sameElements += 1
            # 统计测试中某类数量
            if int(list1[j][0]) == i:
                testElements += 1
            # 统计真值中某类数量
            if int(list2[j]) == i:
                trueElements += 1
        # 避免分母为0
        if testElements == 0:
            testElements = 1
        if trueElements == 0:
            trueElements =1
        # 准确率:一致/测试集  召回率:一致/真值
        precision = sameElements/testElements
        pr_list[i] = precision
        recall = sameElements/trueElements
        re_list[i] = recall
        if (precision+recall) == 0:
            f1score = -1
        else:
            f1score = 2*precision*recall/(precision+recall)
        print("类别 %s\t精确率 %f\t召回率 %f\tF1 score %f" % \
                ([(v,k) for k,v in name_dic.items() if v==i], precision,\
                 recall, f1score))
    # 将所有不一致的标签输出
    inconsistentElements = 0
    # 用于输出文件路径
    #test_path_tmp = 0
    #for i in range(len(list1)):
    #    if int(list1[i][0]) != int(list2[i]):
    #        print("%s was classified wrongly as %s, its path: " % \
    #            ([(v,k) for k,v in name_dic.items() if v==int(list2[i])], \
    #            [(v,k) for k,v in name_dic.items() if v==int(list1[i][0])]), \
    #            end="")
    #        print(test_images[test_path_tmp])
    ##        ...
    #    test_path_tmp += 1
    return pr_list, re_list

# 计算不同类的准确率、精度和召回率,找到最容易出错的一个点,仅支持ak==1
def pre_rec_for_each_by_matrix(matrix, name_dic):
    #paper_one_dict_rev = {5:0, 8:1, 7:2, 9:3, 0:4, 10:5, 1:6, 2:7, 3:8, 4:9, 6:10}
    n = len(matrix)
    sum_all = 0
    sum_right = 0
    for i in range(n):
        #按照论文顺序输出
        #i = paper_one_dict_rev[i]
        # 初始化中间参数
        sameElements = matrix[i][i]
        testElements = 0
        trueElements = 0
        sum_right += matrix[i][i] 
        for j in range(n):
            testElements += matrix[j][i]
            trueElements += matrix[i][j]
            sum_all += matrix[i][j]
        # 避免分母为0
        if testElements == 0:
            testElements = 1
        if trueElements == 0:
            trueElements =1
        # 准确率:一致/测试集  召回率:一致/真值
        precision = sameElements/testElements
        recall = sameElements/trueElements
        if (precision+recall) == 0:
            f1score = -1
        else:
            f1score = 2*precision*recall/(precision+recall)
        print("类别 %s\t精确率 %f\t召回率 %f\tF1 score %f" % \
                ([(v,k) for k,v in name_dic.items() if v==i], precision,\
                 recall, f1score))

    print("总体准确率为:", sum_right/sum_all)



#计算混淆矩阵
def cal_confusion_matrix(list1, list2, name_dic):
    #for i in range(len(list1)):
    #   print(list2[i], end=" ")
    if len(list1)!=len(list2):
        print("计算混淆矩阵函数输入长度不等!")
        return
    # 将tensor转化为list
    list3 = []
    for i in range(len(list2)):
        list3.append(int(list2[i]))
    class_num = len(set(list3))
    confusion_matrix_origin = [[0 for col in range(class_num)] for row in range(class_num)]
    #计算矩阵
    for i in set(list3):
        for j in range(len(list1)):
            if list3[j] == i:
                k = list1[j][0]
                confusion_matrix_origin[i][k] += 1
    #矩阵按照图片顺序排序
    confusion_matrix = [[0 for col in range(class_num)] for row in range(class_num)]
    #paper_one_dict = {0:5, 1:8, 2:7, 3:9, 4:0, 5:10, 6:1, 7:2, 8:3, 9:4, 10:6}
    for i in range(len(confusion_matrix_origin)):
        for j in range(len(confusion_matrix_origin)):
            confusion_matrix[i][j] = confusion_matrix_origin[i][j]
            #confusion_matrix[paper_one_dict[i]][paper_one_dict[j]] = confusion_matrix_origin[i][j]
    #原顺序
    #confusion_matrix = confusion_matrix_origin
    #-------------------------输出矩阵--------------------------
    print("真值/预测", end="")
    for i in range(class_num):      #输出第一行表头
        print("\t" + str(i), end="")
    print()
    for i in range(102):
        print("-", end="")
    print()
    for i in range(class_num):
        print(str(i) + "\t|", end="")
        for j in range(class_num):
            if confusion_matrix[i][j] == 0:
                print("\t", end="")
            else:
                print("\t" + str(confusion_matrix[i][j]), end="")
        print()
    #-----------------------------------------------------------
    #计算准确率召回率F1 score
    '''
    for i in range(class_num):
        rowsum, colsum = sum(confusion_matrix[i]), sum(confusion_matrix[r][i] for r in range(class_num))
        try:
            precision = confusion_matrix[i][i]/float(colsum)
            recall = confusion_matrix[i][i]/float(rowsum)
            print('precision: %s' % precision, 'recall: %s' % recall)
            print('F1 score: %s' % str((2*precision*recall)/(precision+recall)))
        except ZeroDivisionError:
            print('precision: %s' % 0, 'recall: %s' %0)
    '''
    #制热力图
    confusion_matrix = np.array(confusion_matrix)
    label = ["Cultivar {}".format(i) for i in range(1, confusion_matrix.shape[0]+1)]
    #label = ["Cabernet Sauvignon", "Chardonnay", "Ecolly", "Muscat Hamburg", 
    #         "Italia", "Shine-Muscat", "Crimson Seedless", "Ruiduhongyu", 
    #         "Hutai 8", "Wink ", "Moon Drops"]
    plot_confusion_matrix(confusion_matrix, label)
    pre_rec_for_each_by_matrix(confusion_matrix, name_dic)

#绘制热力图
def plot_confusion_matrix(cm, classes, normalize=False, title='', cmap=plt.cm.BuGn):

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.axis("equal")

    ax = plt.gca()
    left, right = plt.xlim()
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")
        

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(j, i, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")
    #图的排版 
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    #保存图像
    plt.savefig('figures/confusion_matrix.png', transparent=True, dpi=800) 
    #展示图像 
    #plt.show()
