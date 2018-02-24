---
title: "GPU accelerated deep learning"
date: 2018-02-12T00:50:16+02:00
draft: false
---
## NGC

NVIDIA GPU Cloud https://ngc.nvidia.com provides containers for libaries such as TensorFlow,caffe/caffe2, torch etc.

https://www.nvidia.com/en-us/gpu-cloud/deep-learning-containers/


In order to pull containers an personal account is required.
After creating an account you can generate your NGC API Key which is used with docker login to gain access to registry.

## Requirements
### Hardware
NVIDIA states that NVIDIA Volta- or Pascal-powered GPUs are supported.

NVIDIA GTX 1000-series cards are Pascal™ architecture based and the current consumer line products.

When using he libraries the libraries state CUDA compute capability 5.2 as requirement.
This would suggest GeForce GTX 900 -series card starting with GTX950 would work (https://developer.nvidia.com/cuda-gpus)
This has not been verified.

The 700-, 600-, 500- series cards have insufficient CUDA capability and will not be usable.


### Software
Appropriate NVIDIA drivers for your GPU.

NVIDIA uses own wrapper on docker engine called "nvidia-docker". Nvidia-docker is available through repository for Ubuntu and CentOS.
(exact versions at https://nvidia.github.io/nvidia-docker/)


## Pull and run container

Login and pull tensorflow container


{{< highlight bash "style=emacs" >}}
[root@localhost ~]#  docker login -u '$oauthtoken' nvcr.io
Password: 
Login Succeeded
[root@localhost ~]# docker pull nvcr.io/nvidia/tensorflow:18.01-py2
:
{{< / highlight >}}



Once pull has finished, run container


{{< highlight bash "style=emacs">}}

[root@localhost ~]# nvidia-docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v /home/docker_shared:/docker_shared nvcr.io/nvidia/tensorflow:18.01-py2
                                                                                                                                                
================
== TensorFlow ==
================

NVIDIA Release 18.01 (build 276323)

Container image Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
Copyright 2017 The TensorFlow Authors.  All rights reserved.

Various files include modifications (c) NVIDIA CORPORATION.  All rights reserved.
NVIDIA modifications are covered by the license terms that apply to the underlying project or file.

root@53fbe21807b4:/workspace# 

{{< / highlight >}}


## Test the library

{{< highlight python "style=emacs">}}
root@53fbe21807b4:/workspace# python
Python 2.7.12 (default, Nov 20 2017, 18:23:56) 
[GCC 5.4.0 20160609] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> 

>>> import tensorflow as tf
>>> 
>>> config = tf.ConfigProto()
>>> config.gpu_options.per_process_gpu_memory_fraction = 0.4
>>> 
>>> with tf.device('/device:GPU:0'):
...   a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
...   b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
...   c = tf.matmul(a, b)
... 
>>> sess = tf.Session(config=config)
2018-02-12 03:03:33.583866: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1050 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
>>> 
>>> print(sess.run(c))
[[ 22.  28.]
 [ 49.  64.]]
>>> 
{{< / highlight >}}



## Performance comparization

### Convolutional Neural Network training

Convolutional Neural Network training was used for performance comparison. https://www.tensorflow.org/tutorials/deep_cnn


#### CPU only

System:

CPU: Intel(R) Xeon(R) CPU E5-2658 0 @ 2.10GHz
Memory: 64G of DDR3 @1333 MHz

{{< highlight bash "style=emacs">}}

[root@53fbe21807b4:/workspace/models/tutorials/image/cifar10]# python cifar10_train.py
Filling queue with 5000 CIFAR images before starting to train. This will take a few minutes.
:
2018-02-16 03:25:08.344427: step 230, loss = 3.72 (595.8 examples/sec; 0.215 sec/batch)
2018-02-16 03:25:10.500006: step 240, loss = 3.78 (593.8 examples/sec; 0.216 sec/batch)
2018-02-16 03:25:12.669232: step 250, loss = 3.71 (590.1 examples/sec; 0.217 sec/batch)
2018-02-16 03:25:14.835404: step 260, loss = 3.51 (590.9 examples/sec; 0.217 sec/batch)
2018-02-16 03:25:16.983990: step 270, loss = 3.64 (595.7 examples/sec; 0.215 sec/batch)
2018-02-16 03:25:19.117562: step 280, loss = 3.85 (599.9 examples/sec; 0.213 sec/batch)
:

{{< / highlight >}}

##### System utilization 
{{< highlight bash "style=emacs" >}}
[root@localhost ~]# top
Threads: 444 total,  18 running, 426 sleeping,   0 stopped,   0 zombie
%Cpu0  : 78.2 us,  4.3 sy,  0.0 ni, 17.2 id,  0.0 wa,  0.0 hi,  0.3 si,  0.0 st
%Cpu1  : 79.1 us,  4.7 sy,  0.0 ni, 16.3 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu2  : 78.9 us,  4.3 sy,  0.0 ni, 16.7 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu3  : 78.5 us,  5.0 sy,  0.0 ni, 16.6 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu4  : 80.1 us,  4.6 sy,  0.0 ni, 15.2 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu5  : 77.7 us,  4.7 sy,  0.0 ni, 17.6 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu6  : 80.6 us,  4.0 sy,  0.0 ni, 15.4 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu7  : 77.7 us,  4.3 sy,  0.0 ni, 17.9 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu8  : 80.4 us,  4.3 sy,  0.0 ni, 15.3 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu9  : 79.8 us,  4.6 sy,  0.0 ni, 15.6 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu10 : 79.5 us,  4.0 sy,  0.0 ni, 16.6 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu11 : 79.6 us,  3.9 sy,  0.0 ni, 16.4 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu12 : 79.7 us,  4.3 sy,  0.0 ni, 16.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu13 : 79.5 us,  4.0 sy,  0.0 ni, 16.6 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu14 : 78.7 us,  4.0 sy,  0.0 ni, 17.3 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu15 : 80.7 us,  4.0 sy,  0.0 ni, 15.3 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
KiB Mem : 65758232 total, 61758036 free,  1593784 used,  2406412 buff/cache
KiB Swap:  1048572 total,  1048572 free,        0 used. 63528240 avail Mem 

  PID USER      PR  NI    VIRT    RES    SHR S %CPU %MEM     TIME+ COMMAND                                              
 4631 root      20   0 96.945g 844540 133552 R 82.1  1.3   2:59.89 python                                               
 4638 root      20   0 96.945g 844540 133552 R 81.1  1.3   2:59.63 python                                               
 4644 root      20   0 96.945g 844540 133552 R 80.8  1.3   2:59.63 python                                               
 4642 root      20   0 96.945g 844540 133552 R 80.5  1.3   2:59.65 python                                               
 4645 root      20   0 96.945g 844540 133552 R 80.5  1.3   3:00.40 python                                               
 4634 root      20   0 96.945g 844540 133552 R 80.1  1.3   3:00.25 python                                               
 4639 root      20   0 96.945g 844540 133552 R 80.1  1.3   2:59.78 python      
:


[root@localhost ~]# nvidia-smi 
Fri Feb 16 05:28:16 2018       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 390.12                 Driver Version: 390.12                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 105...  Off  | 00000000:02:00.0 Off |                  N/A |
| 13%   55C    P8    N/A /  75W |     61MiB /  4039MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

{{< / highlight >}}

CPU is heavily loaded while GPU is not loaded at all. 




#### With single GPU

GPU: NVIDIA GeForce GTX 1050 Ti


{{< highlight bash "style=emacs" >}}
[root@53fbe21807b4:/workspace/models/tutorials/image/cifar10]# python cifar10_train.py
Filling queue with 5000 CIFAR images before starting to train. This will take a few minutes.
2018-02-16 03:30:01.093730: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1050 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
:
2018-02-16 03:33:01.153808: step 2950, loss = 1.47 (4621.0 examples/sec; 0.028 sec/batch)
2018-02-16 03:33:01.426896: step 2960, loss = 1.48 (4687.1 examples/sec; 0.027 sec/batch)
2018-02-16 03:33:01.699991: step 2970, loss = 1.46 (4687.0 examples/sec; 0.027 sec/batch)
2018-02-16 03:33:01.973810: step 2980, loss = 1.21 (4674.7 examples/sec; 0.027 sec/batch)
2018-02-16 03:33:02.246838: step 2990, loss = 1.35 (4688.2 examples/sec; 0.027 sec/batch)

{{< / highlight >}}



##### System utilization 
{{< highlight bash "style=emacs">}}
[root@localhost ~]# top
Threads: 447 total,   1 running, 446 sleeping,   0 stopped,   0 zombie
%Cpu0  : 25.1 us,  6.4 sy,  0.0 ni, 68.5 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu1  : 25.3 us,  6.1 sy,  0.0 ni, 68.6 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu2  : 25.2 us,  6.8 sy,  0.0 ni, 68.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu3  : 24.9 us,  7.1 sy,  0.0 ni, 68.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu4  : 27.1 us,  6.8 sy,  0.0 ni, 66.1 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu5  : 24.7 us,  7.1 sy,  0.0 ni, 68.2 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu6  : 25.2 us,  6.5 sy,  0.0 ni, 68.4 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu7  : 24.4 us,  6.8 sy,  0.0 ni, 68.8 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu8  : 25.6 us,  6.7 sy,  0.0 ni, 67.7 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu9  : 23.4 us,  6.4 sy,  0.0 ni, 70.2 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu10 : 25.7 us,  6.1 sy,  0.0 ni, 68.2 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu11 : 23.9 us,  6.4 sy,  0.0 ni, 69.7 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu12 : 23.5 us,  6.4 sy,  0.0 ni, 70.1 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu13 : 23.7 us,  6.4 sy,  0.0 ni, 69.9 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu14 : 31.6 us,  5.4 sy,  0.0 ni, 63.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu15 : 23.1 us,  6.7 sy,  0.0 ni, 70.2 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
KiB Mem : 65758232 total, 61129944 free,  2134780 used,  2493508 buff/cache
KiB Swap:  1048572 total,  1048572 free,        0 used. 62900216 avail Mem 

  PID USER      PR  NI    VIRT    RES    SHR S %CPU %MEM     TIME+ COMMAND                                              
 5093 root      20   0 99.529g 1.493g 330812 S 23.3  2.4   0:04.04 python                                               
 5106 root      20   0 99.529g 1.493g 330812 S 18.3  2.4   0:03.98 python                                               
 5107 root      20   0 99.529g 1.493g 330812 S 17.6  2.4   0:03.61 python                                               
 5104 root      20   0 99.529g 1.493g 330812 S 17.3  2.4   0:03.73 python                                               
 5094 root      20   0 99.529g 1.493g 330812 S 16.3  2.4   0:03.52 python                                               
 5098 root      20   0 99.529g 1.493g 330812 S 16.3  2.4   0:03.69 python                                               
 5102 root      20   0 99.529g 1.493g 330812 S 16.3  2.4   0:03.58 python                                               
 5095 root      20   0 99.529g 1.493g 330812 S 15.9  2.4   0:03.55 python                                               
 5097 root      20   0 99.529g 1.493g 330812 S 15.9  2.4   0:03.61 python       
:


[root@localhost ~]# nvidia-smi 
Fri Feb 16 05:30:15 2018       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 390.12                 Driver Version: 390.12                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 105...  Off  | 00000000:02:00.0 Off |                  N/A |
| 10%   62C    P0    N/A /  75W |   1881MiB /  4039MiB |     89%      Default |
+-------------------------------+----------------------+----------------------+

{{< / highlight >}}

CPU load is about 1/4th compared to CPU only case, and GPU is 87-90% loaded.




### Multi GPU train

#### Using single GPU

GPU0: NVIDIA GeForce GTX 1050 Ti (pci bus id: 0000:02:00.0)

{{< highlight bash "style=emacs" >}}
[root@53fbe21807b4:/docker_shared/models-master/tutorials/image/cifar10]# python cifar10_multi_gpu_train.py --num_gpus=1
2018-02-24 20:08:07.794224: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1050 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
2018-02-24 20:08:07.794252: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:1) -> (device: 1, name: GeForce GTX 1050 Ti, pci bus id: 0000:03:00.0, compute capability: 6.1)
:
2018-02-24 20:08:11.653682: step 10, loss = 4.60 (5106.7 examples/sec; 0.025 sec/batch)
2018-02-24 20:08:11.903143: step 20, loss = 4.41 (5385.2 examples/sec; 0.024 sec/batch)
2018-02-24 20:08:12.152155: step 30, loss = 4.51 (4889.2 examples/sec; 0.026 sec/batch)
2018-02-24 20:08:12.401452: step 40, loss = 4.34 (5202.6 examples/sec; 0.025 sec/batch)
2018-02-24 20:08:12.652493: step 50, loss = 4.45 (4973.2 examples/sec; 0.026 sec/batch)
2018-02-24 20:08:12.901474: step 60, loss = 4.24 (5174.2 examples/sec; 0.025 sec/batch)

{{< / highlight >}}

{{< highlight bash "hl_lines=4 6 8 10, style=emacs">}}
[root@localhost ~]# nvidia-smi dmon -s pucvmet
# gpu   pwr  temp    sm   mem   enc   dec  mclk  pclk pviol tviol    fb  bar1 sbecc dbecc   pci rxpci txpci
# Idx     W     C     %     %     %     %   MHz   MHz     %  bool    MB    MB  errs  errs  errs  MB/s  MB/s
    0     -    50    91    66     0     0  3504  1784     0     0  3945     2     -     -     0  1732   535
    1     -    47     0     0     0     0  3504  1341     0     0  3741     2     -     -     0     0     0
    0     -    51    94    70     0     0  3504  1784     0     0  3945     2     -     -     0  2117   258
    1     -    47     0     0     0     0  3504  1341     0     0  3741     2     -     -     0     0     0
    0     -    51    59    43     0     0  3504  1784     0     0  3945     2     -     -     0  2638   467
    1     -    47     0     0     0     0  3504  1341     0     0  3741     2     -     -     0     0     0
    0     -    51    91    68     0     0  3504  1771     0     0  3945     2     -     -     0  2040   569
    1     -    47     0     0     0     0  3504  1341     0     0  3741     2     -     -     0     0     0
:
{{< / highlight >}}


{{< highlight bash "hl_lines=10, style=emacs">}}
[root@localhost ~]# nvidia-smi 
Sat Feb 24 22:26:43 2018       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 390.12                 Driver Version: 390.12                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 105...  Off  | 00000000:02:00.0 Off |                  N/A |
| 24%   45C    P0    N/A /  75W |   3945MiB /  4039MiB |     93%      Default |
+-------------------------------+----------------------+----------------------+
|   1  GeForce GTX 105...  Off  | 00000000:03:00.0 Off |                  N/A |
|  0%   41C    P0    N/A /  75W |   3741MiB /  4040MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
{{< / highlight >}}

{{< highlight bash "style=emacs">}}
[root@localhost ~]# top
top - 23:00:00 up  2:34,  2 users,  load average: 12.17, 16.01, 15.60
Threads: 447 total,   2 running, 445 sleeping,   0 stopped,   0 zombie
%Cpu0  : 27.9 us,  9.1 sy,  0.0 ni, 63.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu1  : 28.6 us,  9.1 sy,  0.0 ni, 62.3 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu2  : 29.9 us,  9.2 sy,  0.0 ni, 60.9 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu3  : 28.4 us, 10.1 sy,  0.0 ni, 61.5 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu4  : 26.7 us,  9.5 sy,  0.0 ni, 63.9 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu5  : 27.3 us,  9.2 sy,  0.0 ni, 63.5 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu6  : 26.5 us, 10.9 sy,  0.0 ni, 62.6 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu7  : 35.3 us,  9.8 sy,  0.0 ni, 54.9 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu8  : 28.6 us,  9.8 sy,  0.0 ni, 61.6 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu9  : 25.8 us,  9.5 sy,  0.0 ni, 64.7 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu10 : 27.1 us, 11.2 sy,  0.0 ni, 61.7 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu11 : 27.0 us,  9.5 sy,  0.0 ni, 63.5 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu12 : 27.4 us,  8.4 sy,  0.0 ni, 64.2 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu13 : 26.1 us,  9.2 sy,  0.0 ni, 64.7 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu14 : 26.2 us, 10.7 sy,  0.0 ni, 63.1 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu15 : 27.7 us, 10.5 sy,  0.0 ni, 61.8 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
KiB Mem : 65758232 total, 60845436 free,  2591788 used,  2321008 buff/cache
KiB Swap:  1048572 total,  1048572 free,        0 used. 62446220 avail Mem 

  PID USER      PR  NI    VIRT    RES    SHR S %CPU %MEM     TIME+ COMMAND                                       
 6344 root      20   0 42.383g 1.927g 420316 S 27.2  3.1   0:04.41 python                                        
 6346 root      20   0 42.383g 1.927g 420316 S 25.2  3.1   0:03.37 python                                        
 6337 root      20   0 42.383g 1.927g 420316 S 23.2  3.1   0:03.30 python                                        
 6345 root      20   0 42.383g 1.927g 420316 S 23.2  3.1   0:03.24 python                                        
 6336 root      20   0 42.383g 1.927g 420316 S 22.5  3.1   0:03.31 python                                        
 6340 root      20   0 42.383g 1.927g 420316 S 22.2  3.1   0:03.22 python                                        
 6333 root      20   0 42.383g 1.927g 420316 S 21.9  3.1   0:03.35 python                                        
 6338 root      20   0 42.383g 1.927g 420316 S 21.5  3.1   0:03.38 python                                        
 6335 root      20   0 42.383g 1.927g 420316 S 21.2  3.1   0:03.67 python     
:
{{< / highlight >}}


#### Using two GPUs

GPU0: NVIDIA GeForce GTX 1050 Ti (pci bus id: 0000:02:00.0)

GPU1: NVIDIA GeForce GTX 1050 Ti (pci bus id: 0000:03:00.0)

{{< highlight bash "style=emacs">}}
[root@53fbe21807b4:/docker_shared/models-master/tutorials/image/cifar10]# python cifar10_multi_gpu_train.py --num_gpus=2
2018-02-24 20:13:57.238343: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1050 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
2018-02-24 20:13:57.238370: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:1) -> (device: 1, name: GeForce GTX 1050 Ti, pci bus id: 0000:03:00.0, compute capability: 6.1)
:
2018-02-24 20:14:02.606054: step 10, loss = 4.58 (9662.2 examples/sec; 0.013 sec/batch)
2018-02-24 20:14:02.861361: step 20, loss = 4.91 (10382.4 examples/sec; 0.012 sec/batch)
2018-02-24 20:14:03.114567: step 30, loss = 4.43 (10248.2 examples/sec; 0.012 sec/batch)
2018-02-24 20:14:03.369792: step 40, loss = 4.33 (9618.7 examples/sec; 0.013 sec/batch)
2018-02-24 20:14:03.628529: step 50, loss = 4.33 (10117.7 examples/sec; 0.013 sec/batch)
2018-02-24 20:14:03.883167: step 60, loss = 4.27 (10204.5 examples/sec; 0.013 sec/batch)
2018-02-24 20:14:04.135606: step 70, loss = 4.32 (10136.5 examples/sec; 0.013 sec/batch)
2018-02-24 20:14:04.389014: step 80, loss = 4.17 (10003.9 examples/sec; 0.013 sec/batch)
2018-02-24 20:14:04.645723: step 90, loss = 4.05 (10460.1 examples/sec; 0.012 sec/batch)
:
{{< / highlight >}}


{{< highlight bash "style=emacs">}}
[root@localhost ~]# nvidia-smi dmon -s pucvmet
# gpu   pwr  temp    sm   mem   enc   dec  mclk  pclk pviol tviol    fb  bar1 sbecc dbecc   pci rxpci txpci
# Idx     W     C     %     %     %     %   MHz   MHz     %  bool    MB    MB  errs  errs  errs  MB/s  MB/s
    0     -    47    91    67     0     0  3504  1784     0     0  3945     2     -     -     0  2546   259
    1     -    50    89    68     0     0  3504  1784     0     0  3945     2     -     -     0  1684  1111
    0     -    47    93    70     0     0  3504  1784     0     0  3945     2     -     -     0  2658   492
    1     -    50    91    69     0     0  3504  1784     0     0  3945     2     -     -     0  1939   238
    0     -    47    89    65     0     0  3504  1784     0     0  3945     2     -     -     0  2666   468
    1     -    51    87    65     0     0  3504  1784     0     0  3945     2     -     -     0  1732   385
    0     -    47    46    33     0     0  3504  1784     0     0  3945     2     -     -     0  2589   464
    1     -    50    81    62     0     0  3504  1784     0     0  3945     2     -     -     0  1844   422

{{< / highlight >}}


{{< highlight bash "hl_lines=10 13,style=emacs">}}
root@localhost ~]# nvidia-smi 
Sat Feb 24 22:27:05 2018       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 390.12                 Driver Version: 390.12                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 105...  Off  | 00000000:02:00.0 Off |                  N/A |
| 45%   68C    P0    N/A /  75W |   3945MiB /  4039MiB |     90%      Default |
+-------------------------------+----------------------+----------------------+
|   1  GeForce GTX 105...  Off  | 00000000:03:00.0 Off |                  N/A |
| 17%   64C    P0    N/A /  75W |   3945MiB /  4040MiB |     91%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
{{< / highlight >}}


{{< highlight bash "style=emacs">}}
[root@localhost ~]# top
top - 22:57:55 up  2:32,  2 users,  load average: 15.57, 17.79, 16.01
Threads: 446 total,  28 running, 418 sleeping,   0 stopped,   0 zombie
%Cpu0  : 54.1 us, 16.2 sy,  0.0 ni, 29.7 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu1  : 54.2 us, 16.2 sy,  0.0 ni, 29.6 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu2  : 54.8 us, 16.3 sy,  0.0 ni, 28.9 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu3  : 54.5 us, 15.5 sy,  0.0 ni, 30.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu4  : 51.9 us, 18.9 sy,  0.0 ni, 29.3 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu5  : 52.9 us, 17.6 sy,  0.0 ni, 29.5 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu6  : 55.6 us, 16.3 sy,  0.0 ni, 28.1 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu7  : 60.4 us, 17.8 sy,  0.0 ni, 21.8 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu8  : 52.0 us, 17.4 sy,  0.0 ni, 30.5 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu9  : 53.2 us, 14.6 sy,  0.0 ni, 32.2 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu10 : 53.5 us, 16.2 sy,  0.0 ni, 30.3 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu11 : 53.4 us, 15.4 sy,  0.0 ni, 31.2 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu12 : 53.7 us, 15.4 sy,  0.0 ni, 30.9 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu13 : 51.2 us, 18.7 sy,  0.0 ni, 30.1 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu14 : 52.8 us, 16.4 sy,  0.0 ni, 30.8 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu15 : 53.9 us, 18.5 sy,  0.0 ni, 27.6 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
KiB Mem : 65758232 total, 59722952 free,  3651312 used,  2383968 buff/cache
KiB Swap:  1048572 total,  1048572 free,        0 used. 61397632 avail Mem 

  PID USER      PR  NI    VIRT    RES    SHR S %CPU %MEM     TIME+ COMMAND                                       
 4368 root      20   0 43.690g 2.915g 407932 R 41.7  4.6  11:49.44 python                                        
 4370 root      20   0 43.690g 2.915g 407932 R 41.7  4.6  11:55.50 python                                        
 4363 root      20   0 43.690g 2.915g 407932 R 41.4  4.6  11:51.82 python                                        
 4375 root      20   0 43.690g 2.915g 407932 R 40.7  4.6  11:49.29 python                                        
 4372 root      20   0 43.690g 2.915g 407932 R 40.1  4.6  11:50.87 python                                        
 4364 root      20   0 43.690g 2.915g 407932 S 39.7  4.6  11:55.07 python                                        
 4365 root      20   0 43.690g 2.915g 407932 R 39.1  4.6  11:50.46 python                                        
 4369 root      20   0 43.690g 2.915g 407932 R 38.7  4.6  11:54.22 python      
:
{{< / highlight >}}


#### Conclusion

{{< highlight bash "style=emacs">}}
CPU vs. GPU
CPU    : 599.9  examples/sec; 0.213 sec/batch
GPU    : 4688.2 examples/sec; 0.027 sec/batch

GPU scaling
1 GPU  : 5174.2  examples/sec; 0.025 sec/batch
2 GPUs : 10460.1 examples/sec; 0.012 sec/batch
{{< / highlight >}}
 
Even with single low cost GPU the processing is about 8x faster compared to CPU only.

2 GPUs is 2 times as fast as single GPU.



