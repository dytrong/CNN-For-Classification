# train-cnn-for-classification

本代码依赖pytorch 1.2, opencv 3.1, ubuntu16.04 

安装pytorch 

conda install pytorch-nightly -c pytorch


(1) cd pytorch_model, 训练分类网络模型,模型参数保存在model文件夹中,
    data文件夹为分类数据集,里面包含n个子文件夹,每个子文件代表一个类别.

    python network.py 训练网络

(2) cd ../cpp 主要把python训练好的模型转化成c++能够调用

cl_model.py 这个函数就是把python训练好的模型转化为c++能够调用,模型参数保存在cpp_model中,然后example-app.cpp调用.

(3) 运行c++程序具体步骤,采用cmake编译

  a) cmake 编译

    1) 在端面识别文件夹下,新建一个文件夹 mkdir build

    2) cd build

    3) cmake -DCMAKE_PREFIX_PATH=libtorch的绝对路径 ..

    4) make

   其中libtorch下载网站为https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-latest.zip

  b) 如果cmake编译不报错,在当前文件下,输入

    ./example-app ../model_parameter/alexnet_model.pt

    输出运行结果,模型测试通过




