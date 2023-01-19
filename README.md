# MagicInfer
自研开发的深度学习推理框架，具备基本的算子和雏形、配备Yolo Demo，仅供参考学习！



## 1、开发环境

* 开发平台：Ubuntu 20.04
* 开发语言：GCC/G++ 9.4.0、C++ 17
* 数学库：Armadillo & OpenBlas
* 加速库：OpenMP
* 单元测试：GTest
* 性能测试：Google Benchmark
* 底层框架：PNNX（PyTorch Neural Network Exchange）
* 依赖库：sudo apt install libbenchmark-dev、glog
* glog编译
  * `git clone --recursive https://github.com/google/glog.git`
  * `cd glog && mkdir build && cd build` 
  * `cmake .. && make -j8`
  * `sudo make install`




## 2、目录

### 2.1、include/src

* data： 张量类Tensor初始化和实现方法
* layer：算子的实现、Yolo detect类
* parser：PNNX表达式的解析类
* runtime：计算图结构，解析和运行时相关
* utils：通用类库

### 2.2、test

* 单元测试目录

### 2.3、bench

* google benchmark（包含对MobilenetV3和Resnet18的性能测试）

### 2.4、demo

* yolo infer实例



## 3、支持算子

* AdaptiveAvgPooling
* BatchNorm2D
* Concat
* Convolution
* Expression（抽象语法树）
* Flatten（支持HW维度展平和变形）
* HardSigmoid
* HardSwish
* Linear（矩阵相乘，支持二维Tensor相乘）
* MaxPooling
* ReLU
* Sigmoid
* SiLU
* Softmax
* Upsample
* View（支持HW维度展平和变形）
* ......



## 3、编译运行

*  `sudo chmod 777 compile_run.sh`
* `./compile_run.sh`
* 输出文件夹：build、log、out



## 致谢

* KuiperInfer基础版本 https://github.com/zjhellofss/KuiperInfer
* 推理框架NCNN借鉴代码 https://github.com/Tencent/ncnn

* 优秀的数学库Openblas: https://github.com/xianyi/OpenBLAS

* 优秀的数学库Armadillo: https://arma.sourceforge.net/docs.html

* 参考Caffe框架: https://github.com/BVLC/caffe
