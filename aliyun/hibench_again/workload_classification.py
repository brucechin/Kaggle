import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import


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
def acc(pred, real):
    if(len(pred) == len(real)):
        acc = 0
        for i in range(len(pred)):
            if(pred[i] == real[i]):
                acc+=1

        return float(acc/len(pred))
    else:
        return -1

#categories = np.load("categories.npy")
y_4v8g = np.load("hibench_4v8g_y.npy")
y_4v16g = np.load("hibench_4v16g_y.npy")
y_8v16g = np.load("hibench_8v16g_y.npy")
y_half4v8g = np.load("hibench_half4v8g_y.npy")
y_half4v16g = np.load("hibench_half4v16g_y.npy")
y_half8v16g = np.load("hibench_half8v16g_y.npy")

x_4v8g = dim3_to_dim2(np.load("hibench_4v8g_x.npy"))
x_4v16g = dim3_to_dim2(np.load("hibench_4v16g_x.npy"))
x_8v16g = dim3_to_dim2(np.load("hibench_8v16g_x.npy"))
x_half4v8g = dim3_to_dim2(np.load("hibench_half4v8g_x.npy"))
x_half4v16g = dim3_to_dim2(np.load("hibench_half4v16g_x.npy"))
x_half8v16g = dim3_to_dim2(np.load("hibench_half8v16g_x.npy"))

seq = [i for i in range(len(x_4v8g))]
train = np.concatenate((x_4v8g, y_4v8g), axis = 1)

clf = []
#clf.append(KNeighborsClassifier(n_neighbors=3))
#clf.append(DecisionTreeClassifier()) #分类结果较好
clf.append(RandomForestClassifier()) #分类结果较好
#clf.append(GaussianNB())

methods = ['KNN','DecisionTree',"RandomForest","NaiveBayes"]

def evaluation(ratio):
    x_train, x_val, y_train, y_val = train_test_split(x_4v8g,seq)
    for c in clf:

        c.fit(x_train,y_train)
        err = 0
        for i in range(len(y_val)):
            tmp = abs(ratio[y_val][i] - ratio[c.predict(x_val)][i]) / ratio[y_val][i]
            err += tmp[0]
        print("classification error rate : ",err/len(y_val))

def evaluation_rmse(ratio):
    x_train, x_val, y_train, y_val = train_test_split(x_4v8g,seq)
    for c in clf:

        c.fit(x_train,y_train)
        err = 0
        for i in range(len(y_val)):
            tmp = pow((ratio[y_val][i] - ratio[c.predict(x_val)][i]) / ratio[y_val][i],2)
            err += tmp[0]
        print("classification error rate : ",np.sqrt(err/len(y_val)))

def baseline_evaluation(ratio, k):
    err = 0
    base = y_8v16g / y_half4v8g - 1#用base * 系数作为预测
    for i in range(len(ratio)):
        pred = 1 + base[i][0] * k
        tmp = abs(ratio[i][0] - pred) / ratio[i][0]
        err += tmp
    print(err/len(ratio))

def baseline_evaluation_rmse(ratio, k):
    err = 0
    base = y_8v16g / y_half4v8g - 1  # 用base * 系数作为预测
    for i in range(len(ratio)):
        pred = 1 + base[i][0] * k
        tmp = pow(abs(ratio[i][0] - pred) / ratio[i][0],2)
        err += tmp
    print(np.sqrt(err / len(ratio)))


def xgboost_evaluation(ratio):
    x_train, x_val, y_train, y_val = train_test_split(x_4v8g,seq)



#print(y_8v16g/ y_4v8g)
# list = [y_4v8g,y_4v16g,y_8v16g,y_half4v8g,y_half4v16g,y_half8v16g]
# for l in list:

base = y_half4v8g
baseline_evaluation_rmse(y_4v8g / base,0.33)
baseline_evaluation_rmse(y_8v16g / base,1)
baseline_evaluation_rmse(y_4v16g / base,0.66)
baseline_evaluation_rmse(y_half8v16g / base,0.33)
baseline_evaluation_rmse(y_half4v16g / base,0.167)
baseline_evaluation_rmse(y_half4v8g / base,0)




#在运行队列中等待的进程数
#free MEM
# buffer MEM
# cache MEM
# 每秒读磁盘块数
# 写磁盘块数
# 每秒产生中断次数
# 每秒产生上下文切换次数
# 用户，
# 内核，
# 空闲
# 等待IO分别消耗的cpu circle数目
