import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# 文件的组织形式是house price文件夹下面放house_price.py和input文件夹
# input文件夹下面放的是从https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data下载的train.csv  test.csv  sample_submission.csv 和 data_description.txt 四个文件

# step1 检查源数据集，读入数据，将csv数据转换为DataFrame数据
train_df = pd.read_csv("train.csv",index_col = 0)
test_df = pd.read_csv('test.csv',index_col = 0)


# step2 合并数据，进行数据预处理
prices = pd.DataFrame({'price':train_df['SalePrice'],'log(price+1)':np.log1p(train_df['SalePrice'])})
# ps = prices.hist()
# plt.plot()
# plt.show()

y_train = np.log1p(train_df.pop('SalePrice'))
all_df = pd.concat((train_df,test_df),axis = 0)
# print all_df.shape
# print y_train.head()

# step3 变量转化
all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)

# 把category的变量转变成numerical表达形式
# get_dummies方法可以帮你一键one-hot
all_dummy_df = pd.get_dummies(all_df)

# 处理好numerical变量
# 我们这里用mean填充
mean_cols = all_dummy_df.mean()
all_dummy_df = all_dummy_df.fillna(mean_cols)

# 标准化numerical数据
numeric_cols = all_df.columns[all_df.dtypes != 'object']
numeric_col_means = all_dummy_df.loc[:,numeric_cols].mean()
numeric_col_std = all_dummy_df.loc[:,numeric_cols].std()
all_dummy_df.loc[:,numeric_cols] = (all_dummy_df.loc[:,numeric_cols] - numeric_col_means) / numeric_col_std

# step4 建立模型
# 把数据处理之后，送回训练集和测试集
dummy_train_df = all_dummy_df.loc[train_df.index]
dummy_test_df = all_dummy_df.loc[test_df.index]

# 将DF数据转换成Numpy Array的形式，更好地配合sklearn

X_train = dummy_train_df.values
X_test = dummy_test_df.values
X_1,X_2,y_1,y_2 = train_test_split(X_train,y_train,random_state=0)
y_2 = y_2.values
# Ridge Regression
# alphas = np.logspace(-3,2,50)
# test_scores = []
# for alpha in alphas:
#   clf = Ridge(alpha)
#   test_score = np.sqrt(-cross_val_score(clf,X_train,y_train,cv = 10,scoring = 'neg_mean_squared_error'))
#   test_scores.append(np.mean(test_score))
# plt.plot(alphas,test_scores)
# plt.title('Alpha vs CV Error')
# plt.show()

# random forest
# max_features = [.1,.3,.5,.7,.9,.99]
# test_scores = []
# for max_feat in max_features:
#   clf = RandomForestRegressor(n_estimators = 200,max_features = max_feat)
#   test_score = np.sqrt(-cross_val_score(clf,X_train,y_train,cv = 5,scoring = 'neg_mean_squared_error'))
#   test_scores.append(np.mean(test_score))
# plt.plot(max_features,test_scores)
# plt.title('Max Features vs CV Error')
# plt.show()

# Step 5: ensemble
# 用stacking的思维来汲取两种或者多种模型的优点

ridge = Ridge(alpha = 15)
rf = RandomForestRegressor(n_estimators = 500,max_features = .3)
ridge.fit(X_train,y_train)
rf.fit(X_train,y_train)
y_ridge = np.expm1(ridge.predict(X_test))
y_rf = np.expm1(rf.predict(X_test))

def score(x,y):
    score = 0
    for i in range(len(x)):
        score += abs((x[i] - y[i])/y[i])
    return score
#for  i in range(100):
#    y_final = ( i * y_ridge + (100-i) * y_rf) / 100
#    print('{0} : {1}'.format(i,score(y_final,y_2)))

y_ridge = np.expm1(ridge.predict(X_test))
y_rf = np.expm1(rf.predict(X_test))

y_final = ( 9*y_ridge + y_rf) / 10
# Step 6: 提交结果
submission_df = pd.DataFrame(data = {'Id':test_df.index,'SalePrice':y_final})
submission_df.to_csv('submission.csv',columns = ['Id','SalePrice'],index = False)
