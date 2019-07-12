import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from itertools import chain
'''
    该份代码用于lstm和cnn两种深度学习workload的二分类任务，分类准确率达到100%
'''

def acc(pred, real):
    if(len(pred) == len(real)):
        acc = 0
        for i in range(len(pred)):
            if(pred[i] == real[i]):
                acc+=1

        return float(acc/len(pred))
    else:
        return -1
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

x_4v8g = []
cnn_4v8g = dim3_to_dim2(np.load("clean_cnn_8v16g_x.npy"))
lstm_4v8g = dim3_to_dim2(np.load("lstm_8v16g_x.npy"))
x_4v8g.append(cnn_4v8g)
x_4v8g.append(lstm_4v8g)
x_4v8g = np.concatenate(x_4v8g, axis=0)
categories = []
for i in range(len(cnn_4v8g)):
    categories.append(1)
for i in range(len(lstm_4v8g)):
    categories.append(2)


x_train, x_val, y_train, y_val = train_test_split(x_4v8g,categories)
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train, y_train)
print(clf.predict(x_val))
print(y_val)
print(acc(clf.predict(x_4v8g),categories))
