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
from itertools import chain
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.dummy import DummyClassifier,DummyRegressor

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

categories = np.load("categories.npy")
x_4v8g = np.load("x_4v8g.npy")
x_8v16g = np.load("x_8v16g.npy")
y_4v8g = np.load("y_4v8g.npy")
y_8v16g = np.load("y_8v16g.npy")
x_4v16g = np.load("x_4v16g.npy")
y_4v16g = np.load("y_4v16g.npy")

seq = [i for i in range(len(x_4v8g))]
ratio = y_8v16g / y_4v8g
train = np.concatenate((x_4v8g, y_4v8g), axis = 1)



#x_train, x_val, y_train, y_val = train_test_split(x_4v8g,ratio)

clf = []

clf.append(DummyClassifier())
clf.append(KNeighborsClassifier(n_neighbors=3))
clf.append(DecisionTreeClassifier())
clf.append(RandomForestClassifier(n_estimators=8))
clf.append(GaussianNB())
clf.append(SVC(kernel='rbf', C=100, gamma=0.001, probability=True))
clf.append(GradientBoostingClassifier(n_estimators=200))
clf.append(AdaBoostClassifier(n_estimators=100))
methods = ['dummyClassify','KNN','DecisionTree',"RandomForest","NaiveBayes","SVM","GDBT","AdaBoost"]

test_size_list = [0.3]
for j in range(len(test_size_list)):
    index = 0
    x_train, x_val, y_train, y_val = train_test_split(train, seq, test_size=test_size_list[j])
    for c in clf:
        c.fit(x_train,y_train)
        err = 0
        rmse = 0
        for i in range(len(y_val)):
            tmp = abs(ratio[y_val][i] - ratio[c.predict(x_val)][i]) / ratio[y_val][i]
            err += tmp[0]
            tmp2 = pow(ratio[y_val][i] - ratio[c.predict(x_val)][i] / ratio[y_val][i],2)
            rmse += tmp2[0]
        print(methods[index],err/len(y_val),np.sqrt(rmse / len(y_val)))
        index += 1


