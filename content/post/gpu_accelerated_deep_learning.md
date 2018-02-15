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



