import numpy as np
from sklearn.cluster import KMeans
import os

#用于清洗掉npy文件中所有错误值（全为-1的行）


#terasort,wordcount,bayes,kmeans,nutchindex,pagerank
# category = [1,2,3,4,5,6]
# categories = []
# for j in category:
#     for i in range(20):
#         categories.append(j)
# categories = np.array(categories)

path = "./redis/"
filenames = os.listdir(path)
indexlist_to_delete = []

for file in filenames:
    if(file[-5] == 'x'):
        x = np.load(path + file)
        for i in range(len(x)):
            if (x[i][0][0] == -1):
                indexlist_to_delete.append(i)

tmp = set(indexlist_to_delete)
indexlist_to_delete = list(tmp)
print(indexlist_to_delete)
print(len(indexlist_to_delete))

for file in filenames:
    if(file[-3:] == 'npy' and file[:3] == 'ima'):
        x = np.load(path + file)
        print(file,len(x))
        x = np.delete(x, indexlist_to_delete, axis=0)
        print(len(x))
        np.save(path + "clean_"+file, x)

tmp = [11 for i in range(61)]
np.save(path + 'clean_image_compression_categories.npy',tmp)
