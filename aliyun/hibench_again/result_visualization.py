import matplotlib.pyplot as plt
import numpy as np
#base分别为4v8g,4v16g.8v16g,half4v8g,half4v16g,half8v16g,baseline
mae_err = [0.0916,0.1194,0.1256,0.143,0.116,0.0824,0.301]
rmse_err = [0.176,0.389,0.206,0.332,0.192,0.350,0.619]
machine_config = ['c4.large','g5.xlarge','c5.2xlarge','c4.medium','g5.large','c5.xlarge','baseline']


#针对不同任务流预测的误差
workload_types = ['wordcount','terasort','bayes','kmeans','nutchindexing','pagerank','image compression','mongodb','redis','neural transfer','lstm']
mae_baseline = [0.231]
rmse_baseline = []
mae_err_workloads = [0.02,0.024,0.071,0.24,0.183,0.11,0.035,0.095,0.124,0.131,0.115]
rmse_err_workloads = [0.027,0.045,0.09,0.39,0.27,0.183,0.059,0.112,0.172,0.175,0.157]

#针对不同机器学习算法对比预测误差
mae_err_ml = [0.095,0.105,0.109,0.093,0.320,0.167,0.227,0.314]
rmse_err_ml = [0.326,0.395,0.374,0.381,0.754,0.405,0.553,0.739]
ml_type = ['KNN','DecisionTree','Random Forest','Naive Bayes','SVM','GDBT','AdaBoost','Baseline']
time_ml = [0.53,0.37,0.362,1.41,1.32,26.19,2.96,0]

#针对不同时间颗粒度对预测效果的比较
rmse_err_timeslice=[0.335,0.343,0.367,0.353,0.301,0.324]
mae_err_timeslice=[0.131,0.109,0.111,0.094,0.083,0.092]
timeslice_type = ['1','2','4','8','16','32']



#随机森林里n_estimator(决策树数量)的超参数对预测结果的影响
mae_err_n_estimator = [0.099,0.106,0.111,0.089,0.085,0.079,0.076,0.081,0.071,0.070]
rmse_err_n_estimator =[0.407,0.359,0.210,0.221,0.173,0.181,0.164,0.149,0.156,0.160]
time_estimator = [0.06,0.09,0.15,0.27,0.53,1.07,2.12,4.28,8.71,17.37]
estimator_type = [1,2,4,8,16,32,64,128,256,512]


#随机森林里决策树最大深度超参数对预测结果的影响
mae_err_max_depth = [0.119,0.095,0.101,0.074,0.095,0.075,0.0701,0.0702]
rmse_err_max_depth = [0.239,0.211,0.212,0.191,0.187,0.171,0.141,0.159]
depth_type = [4,8,16,32,64,128,256,512]
time_depth = [0.98,1.24,1.72,2.47,3.32,3.32,3.29,3.28]



#随机森林里max feature超参数对预测结果的影响
mae_err_max_feature = [0.079,0.091,0.086,0.078,0.096,0.081,0.072]
rmse_err_max_feature = [0.188,0.158,0.172,0.197,0.208,0.152,0.160]
feature_type = ['0.05','0.1','0.2','0.4','0.8','log2','sqrt']
time_feature = [3.3,6.1,12.3,24.6,49.7,3.3,1.74]

#随机森林里n_job超参数对训练耗时的影响
time_parallel = [52.3,28.4,24.1,21.4,22.0,21.1,22.1]
parallel_type = [1,2,4,8,16,32,64]



ernest = []
ours = []

for i in range(0,100):
    ernest.append(i*10)
    ours.append(200 + i)

x = [i for i in range(0,100)]
plt.figure()
plt.scatter(x,ernest,edgecolors='blue',marker='o',label="ernest")
plt.scatter(x,ours,edgecolors='red',marker='v',label="ours")
plt.legend()
plt.xlabel("user input tasks")
plt.ylabel("accumulative extra cost")
plt.show()


# xticks_type = parallel_type
# x = range(len(xticks_type))
# plt.figure()
# plt.bar(x,time_parallel)
# #for a,b in zip(x,time_parallel):
# #    plt.text(a,b+0.1,'{}'.format(b),ha='center',va='bottom',fontsize=11)
# plt.xticks(x,xticks_type,rotation=45)
# plt.xlabel('n_job value')
# plt.ylabel('training time(s)')
# plt.title('training time under different parallelism level')
# plt.show()


# plt.figure()
# plt.bar(x,mae_err_max_depth)
# #plt.yticks(np.arange(0.05, 0.13, step=0.01))
# plt.xticks(x,xticks_type,rotation=45)
# plt.axis([-1,8,0.05,0.13])
# plt.xlabel('max depth value')
# plt.ylabel('error')
# plt.title('MAE prediction error')
# plt.show()
#
# plt.figure()
# plt.bar(x,rmse_err_max_depth)
# plt.axis([-1,8,0.12,0.25])
# plt.xticks(x,xticks_type,rotation=45)
# #plt.yticks(np.arange(0.12, 0.45, step=0.02))
# plt.xlabel('max depth value')
# plt.ylabel('error')
# plt.title('RMSE prediction error')
# plt.show()