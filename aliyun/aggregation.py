#这份代码用于把每台机器上跑的所有data 相对应地merge到一起，使得每台机器上每行数据都是同一条任务跑出来地结果

import os
import numpy as np
from itertools import chain
def dim3_to_dim2(arr):
    '''
    输入是三维的arr，需要压成两维度以供后续训练
    :param arr:
    :return:
    '''
    res = []
    for i in range(len(arr)):
        tmp = list(chain(*arr[i]))
        res.append(tmp)
    return np.array(res)
path = "./aggregate_data/"
filenames = os.listdir(path)

x_4v8g = []
x_4v16g = []
x_8v16g = []
y_4v8g = []
y_4v16g = []
y_8v16g = []
categories = []
#x是输入特征，y是程序用时，categories是程序所属类别
#hibench六种任务流用了1，2，3，4，5，6
#mongo为7
#redis为8
#cnn/style transfer为9
#lstm为10
#deep learning compression为11

for file in filenames:
    tmp = np.load(path + file)

    if('4v8g_x' in file):
        x_4v8g.append(dim3_to_dim2(tmp))

        print(file, len(tmp))

    if('4v8g_y' in file):
        y_4v8g.append(tmp)
    if('4v16g_x' in file):
        x_4v16g.append(dim3_to_dim2(tmp))
    if('4v16g_y' in file):
        y_4v16g.append(tmp)
    if('8v16g_x' in file):
        x_8v16g.append(dim3_to_dim2(tmp))
    if('8v16g_y' in file):
        y_8v16g.append(tmp)
    if('categories' in file):
        categories.append(tmp)

#np.append() 会把多维数组压成一维再append？
#记得需要按顺序排列然后合并起来
x_4v8g = np.concatenate(x_4v8g,axis=0)
y_4v8g = np.concatenate(y_4v8g,axis=0)
x_4v16g = np.concatenate(x_4v16g,axis=0)
y_4v16g = np.concatenate(y_4v16g,axis=0)
x_8v16g = np.concatenate(x_8v16g,axis=0)
y_8v16g = np.concatenate(y_8v16g,axis=0)
categories = np.concatenate(categories,axis=0)


np.save("x_4v8g.npy",x_4v8g)
np.save("x_4v16g.npy",x_4v16g)
np.save("x_8v16g.npy",x_8v16g)
np.save("y_4v8g.npy",y_4v8g)
np.save("y_4v16g.npy",y_4v16g)
np.save("y_8v16g.npy",y_8v16g)
np.save("categories",categories)

print(len(x_4v8g) == len(y_4v8g),len(x_4v8g), len(y_4v8g))
print(len(x_8v16g) == len(y_8v16g),len(x_8v16g), len(y_8v16g))
print(len(x_4v16g) == len(y_4v16g),len(x_4v16g), len(y_4v16g))
print(len(categories))
