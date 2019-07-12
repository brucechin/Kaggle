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

from itertools import chain
def dim3_to_dim2(arr):

    res = []
    for i in range(len(arr)):
        tmp = list(chain(*arr[i]))
        res.append(tmp)
    return np.array(res)

def min(arr):
    min = arr[0]
    for i in range(len(arr)):
        if(arr[i] < min):
            min = arr[i]

    return min

def max(arr):
    max = arr[0]
    for i in range(len(arr)):
        if(arr[i] > max):
            max = arr[i]

    return max

y_4v8g = np.load("../mongo/mongo_4v8g_y.npy")  / 1.32
y_4v16g = np.load("../mongo/mongo_4v16g_y.npy") / 1.33
y_8v16g = np.load("../mongo/mongo_8v16g_y.npy") / 1.86
y_half4v8g = np.load("../mongo/mongo_half4v8g_y.npy") / 0.65
y_half4v16g = np.load("../mongo/mongo_half4v16g_y.npy") / 0.66
y_half8v16g = np.load("../mongo/mongo_half8v16g_y.npy") / 0.93


min_cost = np.zeros((len(y_4v8g),1),dtype=float)

for i in range(len(y_4v8g)):
    min_cost[i] = max([y_4v8g[i][0],y_4v16g[i][0],y_8v16g[i][0],y_half4v8g[i][0],y_half4v16g[i][0],y_half8v16g[i][0]])
#print(min_cost)
#print(y_half4v16g)
#print(y_8v16g)
x = [i for i in range(len(min_cost))]


reduced_cost1 = 0
reduced_cost2 = 0
for i in range(len(y_4v8g)):
    reduced_cost1 += (y_half4v8g[i][0] - min_cost[i][0]) / y_half4v8g[i][0]
    reduced_cost2 += (y_8v16g[i][0] - min_cost[i][0]) / y_8v16g[i][0]

print(reduced_cost1/len(min_cost),reduced_cost2/len(min_cost))


plt.figure()
plt.scatter(x,min_cost,marker='v',label='ours')
plt.scatter(x,y_half4v8g,marker='o',label='choose the smallest VM')
plt.scatter(x,y_8v16g,marker='*',label='choose the biggest VM')
plt.title("YCSB+MongoDB performance comparasion")
plt.xlabel("trial index")
plt.ylabel("relative performance under same cost")
plt.legend()
plt.show()