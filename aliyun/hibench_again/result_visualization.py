import matplotlib.pyplot as plt

#base分别为4v8g,4v16g.8v16g,half4v8g,half4v16g,half8v16g,baseline
mae_err = [0.0916,0.1194,0.1256,0.143,0.116,0.0824,0.301]
rmse_err = [0.176,0.389,0.206,0.332,0.192,0.350,0.525]
machine_config = ['c4.large','g5.xlarge','c5.2xlarge','c4.medium','g5.large','c5.xlarge','baseline']
x = range(len(machine_config))
plt.figure()
plt.plot(x,mae_err)
plt.xticks(x,machine_config)
plt.xlabel('reference VM type')
plt.ylabel('error')
plt.title('MAE prediction error')
plt.show()

plt.figure()
plt.plot(x,rmse_err)
plt.xticks(x,machine_config)
plt.xlabel('reference VM type')
plt.ylabel('error')
plt.title('RMSE prediction error')
plt.show()