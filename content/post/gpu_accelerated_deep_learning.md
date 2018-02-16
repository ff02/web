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

Login

{{< highlight go >}}
[root@localhost ~]#  docker login -u '$oauthtoken' nvcr.io
Password: 
Login Succeeded
[root@localhost ~]# 
{{< / highlight >}}

Pull tensoflow container

{{< highlight go >}}
[root@localhost ~]# docker pull nvcr.io/nvidia/tensorflow:18.01-py2
:
{{< / highlight >}}



Run container


{{< highlight bash >}}

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

{{< highlight python >}}

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

{{< highlight python >}}

root@53fbe21807b4:/workspace/models/tutorials/image/cifar10# python cifar10_train.py
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
{{< highlight python >}}
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


{{< highlight python >}}
root@53fbe21807b4:/workspace/models/tutorials/image/cifar10# python cifar10_train.py
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
{{< highlight python >}}
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

#### Conclusion

Even with single low cost GPU the processing is about 8x faster.


