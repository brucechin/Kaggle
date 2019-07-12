import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier
import time
import multiprocessing

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

#categories = np.load("../categories.npy")
y_4v8g = np.load("y_4v8g.npy")
y_4v16g = np.load("y_4v16g.npy")
y_8v16g = np.load("y_8v16g.npy")
y_half4v8g = np.load("y_half4v8g.npy")
y_half4v16g = np.load("y_half4v16g.npy")
y_half8v16g = np.load("y_half8v16g.npy")
x_4v8g = np.load("y_4v8g.npy")
x_4v16g = np.load("y_4v16g.npy")
x_8v16g = np.load("y_8v16g.npy")
x_half4v8g = np.load("y_half4v8g.npy")
x_half4v16g = np.load("y_half4v16g.npy")
x_half8v16g = np.load("y_half8v16g.npy")

seq = [i for i in range(len(x_4v8g))]
train = np.concatenate((x_4v8g, y_4v8g), axis = 1)

clf = []

# clf.append(KNeighborsClassifier(n_neighbors=3))
# clf.append(DecisionTreeClassifier()) #分类结果较好
# clf.append(RandomForestClassifier())
# clf.append(GaussianNB())
# clf.append(SVC())
# clf.append(GradientBoostingClassifier())
# clf.append(AdaBoostClassifier())

# clf.append(RandomForestClassifier(n_estimators=100,max_features=0.05))
# clf.append(RandomForestClassifier(n_estimators=100,max_features=0.1))
# clf.append(RandomForestClassifier(n_estimators=100,max_features=0.2))
# clf.append(RandomForestClassifier(n_estimators=100,max_features=0.4))
# clf.append(RandomForestClassifier(n_estimators=100,max_features=0.8))
# clf.append(RandomForestClassifier(n_estimators=100,max_features='sqrt'))
# clf.append(RandomForestClassifier(n_estimators=100,max_features='log2'))

# clf.append(RandomForestClassifier(n_estimators=1))
# clf.append(RandomForestClassifier(n_estimators=2))
# clf.append(RandomForestClassifier(n_estimators=4))
# clf.append(RandomForestClassifier(n_estimators=8))
# clf.append(RandomForestClassifier(n_estimators=16))
# clf.append(RandomForestClassifier(n_estimators=32))
# clf.append(RandomForestClassifier(n_estimators=64))
# clf.append(RandomForestClassifier(n_estimators=128))
# clf.append(RandomForestClassifier(n_estimators=256))

# clf.append(RandomForestClassifier(n_estimators=128,n_jobs=1))
# clf.append(RandomForestClassifier(n_estimators=128,n_jobs=2))
# clf.append(RandomForestClassifier(n_estimators=128,n_jobs=4))
# clf.append(RandomForestClassifier(n_estimators=128,n_jobs=8))
# clf.append(RandomForestClassifier(n_estimators=128,n_jobs=16))
# clf.append(RandomForestClassifier(n_estimators=128,n_jobs=32))
# clf.append(RandomForestClassifier(n_estimators=128,n_jobs=64))


# clf.append(RandomForestClassifier(n_estimators=100,max_depth=4))
# clf.append(RandomForestClassifier(n_estimators=100,max_depth=8))
# clf.append(RandomForestClassifier(n_estimators=100,max_depth=16))
# clf.append(RandomForestClassifier(n_estimators=100,max_depth=32))
# clf.append(RandomForestClassifier(n_estimators=100,max_depth=64))
# clf.append(RandomForestClassifier(n_estimators=100,max_depth=128))
# clf.append(RandomForestClassifier(n_estimators=100,max_depth=256))
clf.append(RandomForestClassifier(n_estimators=100,max_depth=512))

methods = ['KNN','DecisionTree',"RandomForest","NaiveBayes","dummy","help"]
# st = 600
# ed = 650
# x_train = np.append(x_4v8g[:st], x_4v8g[ed:], axis=0)
# x_val = x_4v8g[st:ed]
# y_train = np.append(seq[:st], seq[ed:], axis=0)
# y_val = seq[st:ed]


mae_record = np.zeros((len(clf),1),dtype=float)
rmse_record = np.zeros((len(clf),1),dtype=float)
time_record = np.zeros((len(clf),1),dtype=float)
def evaluation(ratio):
    x_train, x_val, y_train, y_val = train_test_split(train,seq)
    index =0
    for c in clf:
        st = time.clock()
        c.fit(x_train,y_train)
        err = 0
        for i in range(len(y_val)):
            tmp = abs(ratio[y_val][i] - ratio[c.predict(x_val)][i]) / ratio[y_val][i]
            err += tmp[0]
        ed = time.clock()
        time_record[index] += ed - st
        #print("{} mae : ".format(methods[index]),err/len(y_val))
        #print(mae_record, rmse_record)
        mae_record[index] += err/len(y_val)
        index+=1

def evaluation_rmse(ratio):
    x_train, x_val, y_train, y_val = train_test_split(train,seq)
    index = 0
    for c in clf:
        c.fit(x_train,y_train)
        err = 0
        for i in range(len(y_val)):
            tmp = pow((ratio[y_val][i] - ratio[c.predict(x_val)][i]) / ratio[y_val][i],2)
            err += tmp[0]
        #print("{} rsme : ".format(methods[index]),np.sqrt(err/len(y_val)))
        #print(mae_record, rmse_record)
        rmse_record[index] += np.sqrt(err/len(y_val))
        index+=1

def baseline_evaluation(ratio, k):
    err = 0
    base = y_8v16g / y_half4v8g - 1#用base * 系数作为预测
    for i in range(st,ed):
        pred = 1 + base[i][0] * k
        tmp = abs(ratio[i][0] - pred) / ratio[i][0]
        err += tmp
    print(err/len(ratio))

def baseline_evaluation_rmse(ratio, k):
    err = 0
    base = y_8v16g / y_half4v8g - 1  # 用base * 系数作为预测
    for i in range(st,ed):
        pred = 1 + base[i][0] * k
        tmp = pow(abs(ratio[i][0] - pred) / ratio[i][0],2)
        err += tmp
    print(np.sqrt(err / len(ratio)))


# def xgboost_evaluation(ratio):
#     x_train, x_val, y_train, y_val = train_test_split(x_4v8g,seq)
#     model = XGBClassifier()
#     model.fit(x_train,y_train)
#     pred = model.predict(x_val)
#     pred = [ratio[pred[i]][0] for i in range(len(pred))]
#     real = ratio[y_val]
#     real = [real[i][0] for i in range(len(real))]
#     print(pred,real)

def baseline_mae_all(st,ed):
    target = y_4v16g / y_4v8g
    base = 1
    highest = y_8v16g / y_4v8g
    pred = base + (highest - base) * 0.5
    err = 0
    rsme = 0
    for i in range(st,ed):
        err += abs(pred[i][0] - target[i][0]) / target[i][0]
        rsme += pow((pred[i][0] - target[i][0]) / target[i][0],2)
    print(err / (ed-st))
    print(np.sqrt(rsme/(ed-st)))


list = [y_4v16g,y_8v16g,y_half4v8g,y_half4v16g,y_half8v16g]
base = y_4v8g
# for l in list:
#     evaluation_rmse(l / base)
# for l in list:
#    evaluation(l / base)

#baseline_evaluation(y_4v8g / y_half4v8g,0.33)


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
