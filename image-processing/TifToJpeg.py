'''
Author: Lee Hang
Date  : 2020/06/13

将path_source目录下的tif图片转换为jpg图片
?
'''
import os.path
import cv2
import pathlib
import logging



logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.disable(logging.CRITICAL)
logging.debug("程序测试: ")

path_source = pathlib.Path('./datasets/swedish/train')

img_fold_A = list(path_source.glob('*/*'))
logging.debug('%s' %(img_fold_A))

img_fold_list = [str(path) for path in img_fold_A]
num_img1 = len(img_fold_list)

for i in range(num_img1):
    name_A = img_fold_list[i]
    path_A = pathlib.Path(name_A)
    logging.debug('filename is %s' %(path_A))
    im_A = cv2.imread(name_A, 1)
    file_name_temp = name_A[:-4]
    file_name = name_A.replace('.tif', '.jpg')
    file_name = file_name.replace('/train', '/train_jpg')
    logging.debug(file_name)
    cv2.imwrite(file_name, im_A)
