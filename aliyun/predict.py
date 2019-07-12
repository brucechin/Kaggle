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

# time_mongo_4v8g = np.load("./mongo/clean_mongo_4v8g_y.npy")
# time_mongo_4v16g=np.load("./mongo/clean_mongo_4v16g_y.npy")
# time_mongo_8v16g=np.load("./mongo/clean_mongo_8v16g_y.npy")
# time_redis_4v8g = np.load("./redis/clean_redis_4v8g_y.npy")
# time_redis_4v16g = np.load("./redis/clean_redis_4v16g_y.npy")
# time_redis_8v16g = np.load("./redis/clean_redis_8v16g_y.npy")


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

#用4v8g作为参考机型预测其他机型上目标程序的性能
train = np.concatenate((x_4v8g,y_4v8g),axis=1)

seq = [i for i in range(len(x_4v8g))]
x_train, x_val, y_train, y_val =train_test_split(train, seq,test_size=0.02)
clf = RandomForestClassifier(n_estimators=128,max_depth=100,max_features='sqrt')
clf.fit(x_train,y_train)

ratio = [y_4v8g/y_4v8g, y_4v16g/y_4v8g, y_8v16g/y_4v8g, y_half4v8g/y_4v8g, y_half4v16g/y_4v8g, y_half8v16g/y_4v8g]
cost_per_hour = [1.32, 1.33, 1.86, 0.65, 0.66, 0.93]

def predictfPerfCost(refData, refPerf):
    '''
    输入为二维数组，可以同时预测多个任务的性能与开销

    :param refData: 待预测任务在在参考机型上运行的各项资源使用数据
    :param refPerf: 待预测任务在在参考机型上运行的性能
    :return: 返回了两个数组，前者是待预测任务在所有机型配置上运行的性能（包括参考机型），后者是预计完成该任务所需开销
    '''

    index = clf.predict(refData)
    perfPrediction = []
    costPrediction = []
    for i in range(len(ratio)):
        perfPrediction.append(ratio[i][index][0] * refPerf)
        costPrediction.append(ratio[i][index][0] * refPerf * cost_per_hour[i] / 3600)

    print(perfPrediction)
    print(costPrediction)
    return(perfPrediction, costPrediction)

predictfPerfCost([x_val[9]],[y_val[0]])





# def get_predict_ratio(_train_classes, _val_classes, base_y, target_y):
#     '''
#     :param _train_classes: 训练集聚类后每个任务的分类结果
#     :param _val_classes: 验证集预测的分类结果
#     :param base_y: 已知该任务在某机型下的运行时间
#     :param target_y: 目标机型下已知训练集任务运行时间，用验证集预测用时与实际用时比较误差
#     :return: 预测的目标机型下任务运行时间与base机型下运行时间的比值， 以及其真实值
#     '''
#
#     ratio = target_y / base_y
#     real_ratio_list = ratio[splitter_pos:]
#     predict_ratio_list = []
#     for val_class in _val_classes:
#         indexes = get_same_class_indexes(_train_classes, val_class)
#         predict_ratio = 0
#         for i in indexes:
#             predict_ratio += ratio[i]
#         predict_ratio = predict_ratio / len(indexes)
#         predict_ratio_list.append(predict_ratio)
#
#     return predict_ratio_list/real_ratio_list
#
#
# def get_predict_values(_train_classes, _val_classes, base_y, target_y):
#     '''
#     :param _train_classes: 训练集聚类后每个任务的分类结果
#     :param _val_classes: 验证集预测的分类结果
#     :param base_y: 已知该任务在某机型下的运行时间
#     :param target_y: 目标机型下已知训练集任务运行时间，用验证集预测用时与实际用时比较误差
#     :return: 预测的目标机型下任务运行时间与base机型下运行时间的比值， 以及其真实值
#     '''
#
#     ratio = target_y / base_y
#     real_ratio_list = ratio[splitter_pos:]
#     predict_ratio_list = []
#     for val_class in _val_classes:
#         indexes = get_same_class_indexes(_train_classes, val_class)
#         predict_ratio = 0
#         for i in indexes:
#             predict_ratio += ratio[i]
#         predict_ratio = predict_ratio / len(indexes)
#         predict_ratio_list.append(predict_ratio)
#
#     return predict_ratio_list, real_ratio_list
#
#



