# -*- coding: utf-8 -*-
"""
Created on Mon Apr 01 18:47:37 2019

@author: 0820
"""
import numpy as np
times=16
max_time=50000
features=12
data_x=np.zeros((10000,times,features),dtype=int)-1
tmp=np.zeros((max_time,features),dtype=int)-1
data_y=np.zeros((10000,1),dtype=float)-1

files=["hibench_4v8g.report","hibench_4v16g.report","hibench_8v16g.report"]
num=-1
for file in files:
    f=open(file,'rb')
    label=-1
    while(True):
        line=f.readline()
        line = line.decode("gbk")

        if len(line)==0:
            print("exit")
            break
        
        if len(line)>4 and line[0]=='2' and line[1]=='0' and line[2]=='1' and line[3]=='9':
            #时间戳
            label=0
            num+=1
            continue
    
        if label==-1:
            continue
        
        if line[0]=='H':
            #print(line.split())
            data_y[num]=line.split()[4]
            if label<times:
                for t in range(times):
                    for j in range(12):
                        #print(tmp[t*label/times,j])
                        data_x[num,t,j]=tmp[t*label/times,j]
            else:
                for t in range(times):
                    for j in range(12):
                        print(tmp)
                        #print(np.mean(tmp[t*label/times:(t+1)*label/times,j]))
                        data_x[num,t,j]=np.mean(tmp[t*label/times:(t+1)*label/times,j])
            label=-1
        
        elif line[0]=='p':
            continue
        elif line[1]=='r':
            continue
        else:
            print(label)
            line=line.split()
            tmp[label,0]=int(line[0])
            tmp[label,1]=int(line[3])
            tmp[label,2]=int(line[4])
            tmp[label,3]=int(line[5])
            tmp[label,4]=int(line[8])
            tmp[label,5]=int(line[9])
            tmp[label,6]=int(line[10])
            tmp[label,7]=int(line[11])
            tmp[label,8]=int(line[12])
            tmp[label,9]=int(line[13])
            tmp[label,10]=int(line[14])
            tmp[label,11]=int(line[15])
            label+=1
            if label>=400:
                print('over')
    print(file,"finish:",num+1)
    f.close()
print(data_x[0])
np.save('data_x.npy',data_x[0:num+1])
np.save('data_y.npy',data_y[0:num+1])