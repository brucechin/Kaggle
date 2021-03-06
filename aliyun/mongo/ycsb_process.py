# -*- coding: utf-8 -*-
"""
Created on Mon Apr 01 18:47:37 2019

@author: 0820
"""
import numpy as np
times=8
max_time=1000
features=12


files=["mongo_half8v16g.out",'mongo_half4v16g.out','mongo_half4v8g.out']

for file in files:
    data_x = np.zeros((10000, times, features), dtype=int) - 1
    tmp = np.zeros((max_time, features), dtype=int) - 1
    data_y = np.zeros((10000, 1), dtype=float) - 1
    num = -1
    f=open(file,'r')
    label=-1
    while(True):
        line=f.readline()
        if len(line)==0:
            break
        if len(line)>3 and line[2]=='+':
            num+=1
            
        if len(line)>4 and line[0]=='2' and line[1]=='0' and line[2]=='1' and line[3]=='9' and label==-1 and line.split()[4]=='0':
            label=0
            continue
    
        if label==-1:
            continue
        
        if len(line)>4 and line[0]=='2' and line[1]=='0' and line[2]=='1' and line[3]=='9':
            if line.split()[2]=='0':
                label=-1
                continue
            data_y[num]=float(line.split()[6])
            if label<times:
                for t in range(times):
                    for j in range(12):
                        data_x[num,t,j]=tmp[int(t*label/times),j]
            else:
                for t in range(times):
                    for j in range(12):
                        data_x[num,t,j]=np.mean(tmp[int(t*label/times),j])
            label=-1
        
        elif len(line.split())==17 and line[1]!='r':
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
            if label>=max_time:
                print('over times')
    print(file,"finish:",num+1)
        
    f.close()
    print(num)
    #print(data_x[0])
    print(data_y[0:5])
    np.save(file[:-4] +'_x.npy',data_x[0:num+1])
    np.save(file[:-4] +'_y.npy',data_y[0:num+1])