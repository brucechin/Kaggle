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
from itertools import chain
'''
    该代码用于cnn不同网络结构的分类预测，使用有监督学习
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
# categories =   ['alexnet',
#                 'densenet121', 'densenet169', 'densenet201',
#                 'inceptionv3',
#                 'mobilenetv2_0.75', 'mobilenetv2_1.0',
#                 'resnet101_v2',  'resnet152_v2', 'resnet18_v2',  'resnet34_v2', 'resnet50_v2',
#                 'squeezenet1.1',
#                 'vgg11_bn', 'vgg13_bn',  'vgg16_bn', 'vgg19_bn']

# y_4v8g = []
# with open("cnn_report_4v8g.txt") as f:
#     while(True):
#         line = f.readline()
#         if("img/s" in line):
#             network_type = line.split()[0]
#             print(network_type)
#             for i in range(len(categories)):
#                 if(categories[i] == network_type):
#                     if(i == 0):#alex net
#                         y_4v8g.append(0)
#                     elif(i >= 1 and i <= 3):#densenet
#                         y_4v8g.append(1)
#                     elif(i == 4):#inception
#                         y_4v8g.append(2)
#                     elif(i == 5 or i == 6):#mobile net
#                         y_4v8g.append(3)
#                     elif(i>= 7 and i<=11):#resnet
#                         y_4v8g.append(4)
#                     elif(i == 12):#squeezenet
#                         y_4v8g.append(5)
#                     else:#vggnet
#                         y_4v8g.append(6)
#
#         if(len(line) == 0):
#             break

# time_4v8g = np.load("clean_cnn_4v8g_y.npy")
# time_8v16g = np.load("clean_cnn_8v16g_y.npy")
# ratio = time_8v16g / time_4v8g


x_4v8g = dim3_to_dim2(np.load("clean_cnn_8v16g_x.npy"))
y_4v8g = np.load("clean_cnn_networktype.npy")
time_4v8g = np.load("cnn_4v8g_y.npy")
time_8v16g = np.load("cnn_8v16g_y.npy")
x_train, x_val, y_train, y_val = train_test_split(x_4v8g, y_4v8g,test_size=0.4)

clf = DecisionTreeClassifier() #分类结果较好
#clf = svm.SVC() #分类结果很糟糕
#clf = KNeighborsClassifier(n_neighbors=5)
clf1 = RandomForestClassifier() #分类结果较好


#clf = GaussianNB()
clf.fit(x_train, y_train)
print(acc(clf.predict(x_val),y_val))
clf1.fit(x_train, y_train)
print(acc(clf1.predict(x_val),y_val))












# y_4v8g = []
# with open("cnn_report_4v8g.txt") as f:
#     while(True):
#         line = f.readline()
#         if("img/s" in line):
#             network_type = line.split()[0]
#             print(network_type)
#             for i in range(len(categories)):
#                 if(categories[i] == network_type):
#                     y_4v8g.append(i)
#         if(len(line) == 0):
#             break
#

#
# plt.figure()
# plt.xlabel("workloads")
# plt.ylabel("time used")
# plt.plot(y_4v8g,label="4v8g time",color = 'blue')
# plt.plot(y_8v16g,label="8v16g time",color = 'red')
# plt.legend()
# plt.show()