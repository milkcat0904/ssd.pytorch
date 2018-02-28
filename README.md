# SSD: Single Shot MultiBox Object Detector, in PyTorch
A [PyTorch](http://pytorch.org/) implementation of [Single Shot MultiBox Detector](http://arxiv.org/abs/1512.02325) from the 2016 paper by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang, and Alexander C. Berg.  The official and original Caffe code can be found [here](https://github.com/weiliu89/caffe/tree/ssd). 


<img align="right" src= "https://github.com/amdegroot/ssd.pytorch/blob/master/doc/ssd.png" height = 400/>

### Table of Contents
- <a href='#enironment'>Enironment</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training-ssd'>Train</a>
- <a href='#evaluation'>Evaluate</a>
- <a href='#performance'>Performance</a>
- <a href='#demos'>Demos</a>
- <a href='#todo'>Future Work</a>
- <a href='#references'>Reference</a>

&nbsp;
&nbsp;
&nbsp;
&nbsp;

## Enironment
- 安装或者升级最新版pytorch Install [PyTorch](http://pytorch.org/) 
	python 3以上
- 安装可视化模块visdom(VOC数据集可用)

	We now support [Visdom](https://github.com/facebookresearch/visdom) for real-time loss visualization during training! 
	  
	  First install Python server and client 
	  $ pip install visdom
	  
	  Start the server (probably in a screen or tmux)
	  $ python -m visdom.server
	  
	  Then (during training) navigate to http://localhost:8097/ 
	  
## Datasets
### VOC Dataset
##### Download VOC2007 trainval & test

```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
	$ sh data/scripts/VOC2007.sh DWONLOAD_PATH
下载数据好后会自动删除压缩文件
```

##### Download VOC2012 trainval

```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
	$ sh data/scripts/VOC2012.sh DWONLOAD_PATH
```

## Training SSD
- 下载预训练网络参数

	First download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at:
https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth

	By default, we assume you have downloaded the file in the `ssd.pytorch/weights` dir:

```Shell
	$ mkdir weights
	$ cd weights
	$ wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

- 训练模型

To train SSD using the train script simply specify the parameters listed in `train.py` as a flag or manually change them.

```Shell
	$ python train.py --voc_root DATASET_PATH
```
## Evaluation
To evaluate a trained network:

```Shell
	$ python eval.py --voc_root DATASET_PATH
```

You can specify the parameters listed in the `eval.py` file by flagging them or manually changing them.  


<img align="left" src= "https://github.com/amdegroot/ssd.pytorch/blob/master/doc/detection_examples.png">

## Performance

#### VOC2007 Test

##### mAP

| Original | Converted weiliu89 weights | From scratch w/o data aug | From scratch w/ data aug |
|:-:|:-:|:-:|:-:|
| 77.2 % | 77.26 % | 58.12% | 77.43 % |

##### FPS
**GTX 1060:** ~45.45 FPS 

## Demos

### Use a pre-trained SSD network for detection

#### Download a pre-trained network
- We are trying to provide PyTorch `state_dicts` (dict of weight tensors) of the latest SSD model definitions trained on different datasets.  
- Currently, we provide the following PyTorch models: 
    * SSD300 v2 trained on VOC0712 (newest PyTorch version)
      - https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth
    * SSD300 v2 trained on VOC0712 (original Caffe version)
      - https://s3.amazonaws.com/amdegroot-models/ssd_300_VOC0712.pth
- Our goal is to reproduce this table from the [original paper](http://arxiv.org/abs/1512.02325) 
<p align="left">
<img src="http://www.cs.unc.edu/~wliu/papers/ssd_results.png" alt="SSD results on multiple datasets" width="800px"></p>

### Try the demo notebook
- Make sure you have [jupyter notebook](http://jupyter.readthedocs.io/en/latest/install.html) installed.
- Two alternatives for installing jupyter notebook:
    1. If you installed PyTorch with [conda](https://www.continuum.io/downloads) (recommended), then you should already have it.  (Just  navigate to the ssd.pytorch cloned repo and run): 
    `jupyter notebook` 

    2. If using [pip](https://pypi.python.org/pypi/pip):
    
```Shell
# make sure pip is upgraded
pip3 install --upgrade pip
# install jupyter notebook
pip install jupyter
# Run this inside ssd.pytorch
jupyter notebook
```

- Now navigate to `demo/demo.ipynb` at http://localhost:8888 (by default) and have at it!

