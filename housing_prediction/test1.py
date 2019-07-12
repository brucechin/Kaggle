import matplotlib.pyplot as plt
import numpy as np


y_8 = [0.044,0.043,0.031,0.021,0.017,0.012]
y_4 = [0.021,0.032,0.015,0.011,0.013,0.002]
y_2 = [0.017,0.013,0.009,0.010,0.005,0.001]

plt.figure()
plt.plot(y_8,color='r',marker='*',label = 'GPU group size 8')
plt.plot(y_4,color='b',marker='d',label = 'GPU group size 4')
plt.plot(y_2,color='g',marker='s',label = 'GPU group size 2')
plt.ylabel('Gain accuracy from global BN')
plt.xlabel('Batch size')
ax = plt.gca()
ax.set_xticklabels(['0','8','16','32','64','128','256'])
plt.show()