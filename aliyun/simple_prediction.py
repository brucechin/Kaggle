#这段代码最初用于做简单的预测算法，现在已废弃

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
def vector_divide(list1, list2):
    res = 0
    if(len(list1) == len(list2)):
        for i in range(len(list1)):
            res += list1[i] / list2[i]
    return float(res/len(list1))


small_file = "2v_4g_report.csv"
large_file = "4v_8g_report.csv"

small_tmp = pd.read_csv(small_file,names=['workload','throughput','free','buf','cache','si','so','bi','bo','in','cs','us','sy'])
large_tmp = pd.read_csv(large_file,names=['workload','throughput','free','buf','cache','si','so','bi','bo','in','cs','us','sy'])


small = small_tmp[small_tmp['workload'].isin(large_tmp['workload'])]
large = large_tmp[large_tmp['workload'].isin(small_tmp['workload'])]

features = small.ix[:,2:]


y_pred = KMeans(n_clusters=10).fit_predict(features)



def err_cal(row_id):
    work_type = y_pred[row_id]
    same_type_ids = []
    for i in range(len(y_pred)):
        if(y_pred[i] == work_type and i != row_id):
            same_type_ids.append(i)
    #print(row_id, same_type_ids)
    #for i in same_type_ids:
    same_type_workload = small.iloc[same_type_ids]['workload']

    small_throughput = list(small[small['workload'].isin(same_type_workload)]['throughput'])
    large_throughput = list(large[large['workload'].isin(same_type_workload)]['throughput'])
    # print(small_throughput)
    # print(large_throughput)
    multiply = vector_divide(large_throughput,small_throughput)
    real_value = large[large['workload'] == small.iloc[row_id]['workload']]['throughput']
    predict_value = small.iloc[row_id]['throughput'] * multiply
    dummy_predict_value = small.iloc[row_id]['throughput'] * 2

    err = (real_value - predict_value)/real_value
    dummy_err = (real_value - dummy_predict_value)/real_value
    return abs(err.iloc[0]), abs(dummy_err.iloc[0])


errors = []
dummy_errors = []

for i in range(5,75):
    err,dummy_err = err_cal(i)
    errors.append(err)
    dummy_errors.append(dummy_err)

print(errors)
print(dummy_errors)
x = [i for i in range(70)]


plt.figure()
plt.title("2v4g predict 4v8g throughput error rate")
plt.xlabel("workloads")
plt.ylabel("error rate")
plt.plot(x,errors,label="predict",color = 'blue')
plt.plot(x,dummy_errors,label="dummy",color = 'red')
plt.legend()
plt.show()