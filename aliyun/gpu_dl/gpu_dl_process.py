#encoding=gbk
import numpy as np
import random
import matplotlib.pyplot as plt
times=8
max_time=100000
features=21
data_x=np.zeros((100000,times,features),dtype=int)-1
tmp=np.zeros((max_time,features),dtype=int)-1
data_y=np.zeros((100000,1),dtype=float)-1

gpu_file = 'gtx1080_gpu2.report'
cpu_file = 'gtx10802.report'
perf_file = 'gtx1080_perf2.report'

def clean_string(string):
    '''
    主要是去除逗号，在perf文件中使用
    :param string: 待清洗的string
    :return:
    '''
    rst = ""
    for i in string:
        if(i == ','):
            continue
        rst += i
    return rst

def read_gpu_file(filename,output_filename):
    gpu_data = np.zeros((100000, 4), dtype=int) - 1
    label = 0
    g = open(filename,'rb')
    while(True):
        gpu_line = g.readline()
        gpu_line = gpu_line.decode()

        while("Default" not in gpu_line and len(gpu_line) > 0):
            gpu_line = g.readline()
            gpu_line = gpu_line.decode()

        if (len(gpu_line) == 0):
            break
        gpu_line = gpu_line.split()
        #[1]是风扇强度，[4]是功率，需要把单位去掉转成int，[8]是消耗的显存，[12]是GPU利用率
        #print(gpu_line[1][:-1],gpu_line[4][:-1],gpu_line[8][:-3],gpu_line[12][:-1])
        gpu_data[label,0] = int(gpu_line[1][:-1])
        gpu_data[label,1] = int(gpu_line[4][:-1])
        gpu_data[label,2] = int(gpu_line[8][:-3])
        gpu_data[label,3] = int(gpu_line[12][:-1])
        label += 1
    print("gpu file length {}".format(label))
    np.save(output_filename,gpu_data[:label])

def read_perf_file(filename,output_filename):
    perf_data = np.zeros((100000, 5),dtype=float) - 1
    label = -1
    p = open(filename,'r')
    for line in p:
        if('started' in line):
            label += 1

        if('page-faults' in line):
            perf_data[label, 0] = int(clean_string(line.split()[0]))

        if('branch-misses' in line):
            perf_data[label, 1] = int(clean_string(line.split()[0]))

        if('L1-dcache-load-misses' in line):
            perf_data[label, 2] = int(clean_string(line.split()[0]))

        if('LLC-load-misses' in line):
            perf_data[label, 3] = int(clean_string(line.split()[0]))

        if('cpu-migrations' in line):
            perf_data[label, 4] = int(clean_string(line.split()[0]))
    print(label)
    #print(perf_data[:label])
    np.save(output_filename,perf_data[:label+1])


def read_cpu_file(filename,gpu_filename,perf_filename):
    '''
    读入gpu以外的其他硬件利用率数据，将上一步gpu和perf数据处理的也打开，一起写入outputfile
    :param filename:
    :param output_filename:
    :return:
    '''
    c = open(filename, 'rb')
    gpu_data = np.load(gpu_filename)
    perf_data = np.load(perf_filename)
    global_counter = 0
    prev_index = 0 #本段任务起始的index
    label = 0#这个任务执行了多少秒（因为收集数据是一秒一次
    num = 0#parse到第几个任务了
    while(True):
        line=c.readline()
        line = line.decode()
        if len(line)==0:
            break
        if(num == 50):
            break

        if ("FINISH" in line):

            data_y[num] = label * 1
            if label <= times:  # 太短的任务数据忽略掉了
                for t in range(times):
                    for j in range(12):
                        data_x[num, t, j] = tmp[int(t * label / times), j]
                    for k in range(12,16):
                        data_x[num, t, k] = gpu_data[prev_index + int((t * label) / times), k-12]
            else:
                for t in range(times):
                    for j in range(12):
                        data_x[num, t, j] = tmp[int(t * label / times), j]
                    for k in range(12,16):
                        data_x[num, t, k] = gpu_data[prev_index + int((t * label) / times), k-12]

                    for n in range(16,21):
                        data_x[num, t, n] = perf_data[prev_index + int((t * label) / times), n - 16]

            label = 0
            num += 1
            prev_index = global_counter

        if (len(line.split()) > 15 and line[1] != 'r' ):
            global_counter += 1
            line = line.split()
            tmp[label, 0] = int(line[0])
            tmp[label, 1] = int(line[3])
            tmp[label, 2] = int(line[4])
            tmp[label, 3] = int(line[5])
            tmp[label, 4] = int(line[8])
            tmp[label, 5] = int(line[9])
            tmp[label, 6] = int(line[10])
            tmp[label, 7] = int(line[11])
            tmp[label, 8] = int(line[12])
            tmp[label, 9] = int(line[13])
            tmp[label, 10] = int(line[14])
            tmp[label, 11] = int(line[15])
            label += 1
            if label >= max_time:
                print('over times')
    print(num)
    print(data_x[:num])
    print(data_y[:num])
    print(label)
    print(global_counter)
    np.save("gtx1080_x_2.npy",data_x[:num])
    np.save("gtx1080_y_2.npy",data_y[:num])

read_perf_file(perf_file,"perf.npy")
read_gpu_file(gpu_file,"gpu.npy")
read_cpu_file(cpu_file,"gpu.npy","perf.npy")

#gpu_data = np.load("gpu.npy")

# plt.figure()
# plt.title("memory consumption")
# plt.plot(gpu_data[:4000,2])
# plt.show()
#
# plt.figure()
# plt.plot(gpu_data[:4000,1],label = "gpu power")
# plt.plot(gpu_data[:4000,0],label = "fan perf")
# plt.plot(gpu_data[:4000,3],label = "gpu utilization")
# plt.legend()
# plt.show()































#不是都用了相同版本的cudnn，但都是比较接近版本的
#alexnet,inception,vgg16,vgg19,resnet18,resnet34,resnet50,resnet101,resnet152
# gtx1080ti = [13.89,36.87,128,148,32.78,51.31,101.21,154.26,215.54]
# pascaltitanx=[14.56,39.14,128,147,31.54,51.59,103.58,156.44,217.91]
# gtx1080 =[20.74,56.16,182,210,43.94,72.09,149.82,225.80,314.30]
# maxwelltitanx=[25.33,61.98,192,225,51.42,80.23,159.63,247.49,345.45]
#
# comp = [10.6,10.16,8.23,6.14]
#
# xeon=[8495,9849,2195.78,3965.21,6627.25,11306.24,16872.78]
#
# gtx1080ti = np.array(gtx1080ti)
# gtx1080 = np.array(gtx1080)
# pascaltitanx = np.array(pascaltitanx)
# maxwelltitanx = np.array(maxwelltitanx)

# plt.figure()
# plt.plot(gtx1080ti,label="1080ti")
# plt.plot(gtx1080,label="1080")
# plt.plot(maxwelltitanx,label="maxwell titan x")
# plt.plot(pascaltitanx,label="pascal titan x")
# plt.legend()
# plt.show()

# plt.figure()
# plt.title("training time * GFLOPS of GPUs graph")
# plt.plot(gtx1080ti * comp[0],label="1080ti")
# plt.plot(gtx1080 * comp[2],label="1080")
# plt.plot(maxwelltitanx * comp[3],label="maxwell titan x")
# plt.plot(pascaltitanx * comp[1],label="pascal titan x")
# plt.legend()
# plt.show()