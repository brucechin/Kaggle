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

categories = np.load("categories.npy")
# x_4v8g = np.load("x_4v8g.npy")
# x_8v16g = np.load("x_8v16g.npy")
# x_4v16g = np.load("x_4v16g.npy")

y_4v8g = np.load("gpu_dl/gtx1080_y_2.npy")
y_4v16g = np.load("gpu_dl/titan_y_2.npy")
y_8v16g = np.load("image_compression/clean_image_compression_8v16g_y.npy")

def clean_outliner(input, lower, upper):
    for i in range(len(input)):
        if(input[i] < lower or input[i] > upper):
            input[i] = input[i-1]
    return input

y_4v8g = clean_outliner(y_4v8g,0,60000)
y_4v16g = clean_outliner(y_4v16g,0,60000)
y_8v16g = clean_outliner(y_8v16g,0,60000)

x = [i for i in range(len(y_4v8g))]
plt.figure()
plt.title("ResNet training time at two GPUs")
plt.ylabel("run time(s)")
plt.xlabel("trial index")
plt.scatter(x,y_4v16g,label="titan(pascal)")
#plt.scatter(x,y_8v16g,label="c5.2xlarge")
plt.scatter(x,y_4v8g,label="gtx 1080ti")
plt.legend()
plt.show()

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


#
# data_4v8g = pd.DataFrame([("4v8g",float(j)) for j in y_4v8g if(j > 0)],columns=['configuration','performance'])
# data_4v16g = pd.DataFrame([("4v16g",float(j)) for j in y_4v16g if(j > 0)],columns=['configuration','performance'])
# data_8v16g = pd.DataFrame([("8v16g",float(j)) for j in y_8v16g if(j > 0)],columns=['configuration','performance'])
#
#
# print(data_4v8g)
# print(data_8v16g)
# data = pd.concat([data_4v8g,data_4v16g,data_8v16g])
#
#
# ax = sns.violinplot(data = data,x = data['configuration'], y = data['performance'])
# plt.show()




