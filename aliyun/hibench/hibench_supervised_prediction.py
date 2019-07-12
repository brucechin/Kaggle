import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
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

categories = np.load("clean_hibench_categories.npy")
print(categories)
x_4v8g = dim3_to_dim2(np.load("clean_hibench_4v8g_x.npy"))
x_4v16g = dim3_to_dim2(np.load("clean_hibench_4v16g_x.npy"))
x_8v16g = dim3_to_dim2(np.load("clean_hibench_8v16g_x.npy"))


x_train, x_val, y_train, y_val = train_test_split(x_4v8g,categories)
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(x_train, y_train)
print(clf.predict(x_val))
print(y_val)
