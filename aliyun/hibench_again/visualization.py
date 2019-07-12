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
import seaborn as sns
#terasort,wordcount,bayes,kmeans,nutchindex,pagerank 1,2,3,4,5,6
#mongo 7
#redis 8
#cnn/style transfer 9
#lstm 10
#deep learning compression 11

#categories = np.load("categories.npy")
# x_4v8g = np.load("x_4v8g.npy")
# x_8v16g = np.load("x_8v16g.npy")
# x_4v16g = np.load("x_4v16g.npy")

y_4v8g = np.load("hibench_4v8g_y.npy")
y_4v16g = np.load("hibench_4v16g_y.npy")
y_8v16g = np.load("hibench_8v16g_y.npy")
y_half4v8g = np.load("hibench_half4v8g_y.npy")
y_half4v16g = np.load("hibench_half4v16g_y.npy")
y_half8v16g = np.load("hibench_half8v16g_y.npy")

def draw(st=0, ed=296):
    x = [i for i in range(ed-st)]
    plt.figure()
    plt.scatter(x,y_4v8g[st:ed],label="c4.large")
    plt.scatter(x,y_8v16g[st:ed],label="c5.2xlarge")
    plt.scatter(x,y_4v16g[st:ed],label="g5.xlarge")
    plt.scatter(x,y_half4v8g[st:ed],label="c4.medium")
    plt.scatter(x,y_half4v16g[st:ed],label="g5.large")
    plt.scatter(x,y_half8v16g[st:ed],label="c4.xlarge")
    plt.ylabel("run time(s)")
    plt.xlabel("trial index(0-49 is Bayes,50-99 is KMeans)")
    plt.legend()
    plt.show()

draw(100,199)

# index_to_delete = []
# for i in range(len(ratio)):
#     if(ratio2[i] < 0.3 or ratio2[i] > 5 or ratio[i] < 0.3 or ratio[i] > 5):
#         print(categories[i])
#         index_to_delete.append(i)
#
# print(index_to_delete)

# categories = np.delete(categories,index_to_delete)
# x_4v8g = np.delete(x_4v8g,index_to_delete,axis=0)
# x_8v16g = np.delete(x_8v16g,index_to_delete,axis=0)
# y_4v8g = np.delete(y_4v8g,index_to_delete,axis=0)
# y_8v16g = np.delete(y_8v16g,index_to_delete,axis=0)
# x_4v16g = np.delete(x_4v16g,index_to_delete,axis=0)
# y_4v16g = np.delete(y_4v16g,index_to_delete,axis=0)
#
# np.save("x_4v8g.npy",x_4v8g)
# np.save("x_4v16g.npy",x_4v16g)
# np.save("x_8v16g.npy",x_8v16g)
# np.save("y_4v8g.npy",y_4v8g)
# np.save("y_4v16g.npy",y_4v16g)
# np.save("y_8v16g.npy",y_8v16g)
# np.save("categories",categories)



data_4v8g = pd.DataFrame([("4v8g",float(j)) for j in y_4v8g if(j > 0)],columns=['configuration','performance'])
data_4v16g = pd.DataFrame([("4v16g",float(j)) for j in y_4v16g if(j > 0)],columns=['configuration','performance'])
data_8v16g = pd.DataFrame([("8v16g",float(j)) for j in y_8v16g if(j > 0)],columns=['configuration','performance'])


print(data_4v8g)
print(data_8v16g)
data = pd.concat([data_4v8g,data_4v16g,data_8v16g])


#ax = sns.violinplot(data = data,x = data['configuration'], y = data['performance'])
#plt.show()




# plt.figure()
# plt.ylabel("workload time consumed ratio")
# plt.xlabel("workload classes")
# plt.scatter(categories,y_4v16g / y_4v8g,edgecolors="blue")
# plt.scatter(categories,y_8v16g / y_4v8g)
# plt.legend()
# plt.show()