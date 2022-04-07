'''
Author: Lee Hang
Date  : 2022/01/03

将指定文件夹下的数据随机按照4:1化为K折交叉,存放在另一文件夹下
'''

import os
import shutil
import random

def mycopyfile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print (srcfile + " not exist!")
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.copyfile(srcfile,dstfile)      #复制文件
        print ("copy " + srcfile + "-> " + dstfile)

def main():
    # 源数据路径
    file_path = "./exp9/"
    fold = 5
    grape_name_list = []
    count_num = 0
    for root, dirs, files in os.walk(file_path):
        if root == file_path:
            grape_name_list = dirs
        else:
            grape_name = grape_name_list[count_num]
            count_num += 1
            curlist = files
            random.shuffle(curlist)
            dst_path = []
            for j in range(fold):
                # 拷贝的k折目的地
                dst_path.append("./crossfold5/segdata" + str(j+1))
                lfold = len(curlist) / fold
                for k in range(len(curlist)):
                    if lfold*j <= k < lfold*(j+1):
                        mycopyfile(root + "/" + curlist[k], dst_path[j] + "/test/" + grape_name + "/" + curlist[k])
                    else:
                        mycopyfile(root + "/" + curlist[k], dst_path[j] + "/train/" + grape_name + "/" + curlist[k])

if __name__ == "__main__":
    main()
    #for root, dirs, files in os.walk("./segdata/"):
    #    print(type(files))
