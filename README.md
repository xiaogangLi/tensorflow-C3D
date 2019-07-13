# Action Recognition
This is the implementation of 3-dimensional convolutional networks (C3D) approach for Action Recognition in Tensorflow. It contains complete code for preprocessing, training and test. Besides, this repository is easy-to-use and can be developed on Linux and Windows.  

[C3D : Tran, Du , et al. "Learning Spatiotemporal Features with 3D Convolutional Networks." (2015).](https://arxiv.org/abs/1412.0767)

## Getting Started
### 1 Prerequisites  
* Python3.6  
* Tensorflow  
* Opencv-python  
* Pandas  

### 2 Download this repo and unzip it  
`cd ../C3D/Label_Map`  
Open the `label.txt` and revise its class names as yours.  

### 3 Generate directory  
`cd ../C3D/Code`  
`run python make_dir.py`  
Then some subfolders will be generated in `../C3D/Raw_Data` and `../C3D/Data`, where name of the subfolders is your class names defined in `label.txt`.  

### 4 Prepare video clips for training  
According to the class, copy your raw videos to subfolders in ../C3D/Raw_Data. Optionally, you can use the public HMDB-51 dataset, which can be found [here](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/).  
`cd ../C3D/Code`  
`run python prepare_clips.py`  
Clips generated will be saved in the subfolders in `../C3D/Data`.  

### 5 Compute the mean image from training clips(the original paper without this step)  
`cd ../C3D/Code`  
`run python mean_img.py`    
And then a mean image is saved in directory `../C3D/Data`.  

### 6 Training model  
`cd ../C3D/Code`  
`run python train.py PB` or `python train.py CHECKPOINT`  
The model will be saved in directory `../C3D/Model`, where "PB" and "CHECKPOINT" is two ways used for saving model for Tensorflow.  
 
### 7 Test model  
According to Step 4,preparing video clips for test.  
`cd ../C3D/Code`  
`run python test.py N`  
Where N is not more than the number of clips in test set. Note we do not use batch during test. There is out of memory errors with a large N. In this case, you can modify the `test.py` to use batch.    

### 8 Visualize using Tensorboard  
`cd ../C3D`  
`run tensorboard --logdir=Model/`   
Open the URL in browser to visualize model.    

## Other Implementations
[tensorflow-VTN](https://github.com/xiaogangLi/tensorflow-VTN)


## 使用方法  

### 1、安装环境依赖项  
 ① Python3.6  
 ② Tensorflow  
 ③ Opencv-python  
 ④ Pandas  

### 2、下载这个工程到任意目录并解压  
① 切换到目录 `../C3D/Label_Map`,打开`label.txt`,将其中已有的类名修改为你的类名和对应的id。  

### 3、创建保存数据的目录  
① 切换到目录 `../C3D/Code`,然后运行：`python make_dir.py`，在目录`../C3D/Raw_Data` 和`../C3D/Data` 中将会创建子文件夹，文件夹名字为你的类名。  

### 4、准备数据，生成视频片段（clips）  
① 根据类别，将你自己收集到的原始视频数据复制到目录 `../C3D/Raw_Data` 中对应的文件夹中。  
② 切换到目录 `../C3D/Code`, 然后运行：`python prepare_clips.py`，每个类生成的视频片段将会保存在`../C3D/Data` 的子文件夹中，将被用于训练模型。  

### 5、计算训练集的均值图像  
① 切换到目录 `../C3D/Code`,然后运行：`python mean_img.py`，生成的均值图像将会保存在`../C3D/Data` 目录下。  
注：训练时，视频片段中每一帧图像将会被移除均值图像（原论文中并没有这一步预处理）。  

### 6、训练模型  
① 切换到目录 `../C3D/Code`，然后运行`python train.py PB` 或者 `python train.py CHECKPOINT`,参数 "PB" 和 "CHECKPOINT"分别对应Tensorflow保存模型的两种方式。模型保存在 `../C3D/Model`中。  

### 7、测试模型（使用PB模型）  
① 根据步骤4，生成测试数据集的视频片段，同样也保存在`../C3D/Data`的子文件夹中。  
② 切换到目录 `../C3D/Code`，然后运行`python test.py N`,这里N为小于等于测试集中clip的数量的正整数。  
注：由于在测试集上测试时，并没有把测试集划分成多个batch来测试，如果一次性把测试集读入内存，内存可能不够。此时需要进一步修改`test.py`来实现批量测试。

### 8、Tensorflow可视化模型  
① 切换到目录 `../C3D/`，执行：`tensorboard --logdir=Model/`，然后将显示的网站复制到浏览器中打开，可查看模型结构。  
  

## 其他版本
[tensorflow-VTN](https://github.com/xiaogangLi/tensorflow-VTN)

