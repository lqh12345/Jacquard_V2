![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)
# Contents
- [Heading One](#heading-one)
- [Heading Two](#heading-two)
	- [AAA](#aaa)
	- [bbb](#bbb)


# Toolbox for Jacquard V2 Dataset

## Introduction

Jacquard V2 is a dataset for robot vision grasping tasks, which is an enhanced version of the Jacquard dataset. It consists of 11K target objects and 51K images. All images have the RGB-D modality. The annotations include multiple gripper sizes, multiple grasps per image, and grasp locations.

For more details and demos about Jacquard V2 dataset, please refer to _**[[Paper]](https://xxxxxxxx).**_

<span id=download>Please download the dataset</span> through _**[[One Drive]](https://xxxxxxx)**_ or _**[[Baidu Netdisk]](https://pan.baidu.com/s/14SIj1jGyMdYjmKPWMI056Q?pwd=1234)**_.

### Errors present in JacquardV1
The following are three types of problems that exist in Jacquard V1.

<img src="https://github.com/lqh12345/Jacquard_V2/blob/main/figure/Errors%20present%20in%20JacquardV1.png" width="50%">


### Refining Datasets using the Human In the Loop Data Correction Method
<img src="https://github.com/lqh12345/Jacquard_V2/blob/main/figure/Overall%20flowchart.png">

### Jacquard V2 (ours) compared to Jacquard V1
Among them, the green boxes represent the data in the Jacquard V1 dataset, and the red boxes are the additions we made in our Jacquard V2.

<img src="https://github.com/lqh12345/Jacquard_V2/blob/main/figure/JacquardV1_VS_JacquardV2.png" width="60%">


## Quick Start for Jacquard V2

To get started, follow the instructions in this section. We will introduce the simple steps and how you can customize the configuration.&#x20;

### Step 1

#### Dependencies

Please make sure you have installed the following dependencies before using Jacquard V2 dataset.&#x20;
* Python 3+ distribution
* Python requirements can be found in `requirements.txt`.

Quick installation of depedencies

```
pip install -r requirements.txt
```

### Step 2

Once the environment is built successfully, download the [dataset](#download);

After unziping all four parts, the dataset directory structure should be as follows.&#x20;

#### Dataset Directory Structure

```
${DATASET_ROOT}
|-- JacquardV2_Dataset_0
|   |-- 1a1ec1cfe633adcdebbf11b1629fc16a
|   |   |-- 0_1a1ec1cfe633adcdebbf11b1629fc16a_grasps.txt
|   |   |-- 0_1a1ec1cfe633adcdebbf11b1629fc16a_mask.png
|   |   |-- 0_1a1ec1cfe633adcdebbf11b1629fc16a_perfect_depth.tiff
|   |   |-- 0_1a1ec1cfe633adcdebbf11b1629fc16a_RGB.png
|   |   |-- 0_1a1ec1cfe633adcdebbf11b1629fc16a_stereo_depth.tiff
|   |   |-- 1_1a1ec1cfe633adcdebbf11b1629fc16a_grasps.txt
|   |   |-- 1_1a1ec1cfe633adcdebbf11b1629fc16a_mask.png
|   |   |-- 1_1a1ec1cfe633adcdebbf11b1629fc16a_perfect_depth.tiff
|   |   |-- 1_1a1ec1cfe633adcdebbf11b1629fc16a_RGB.png
|   |   |-- 1_1a1ec1cfe633adcdebbf11b1629fc16a_stereo_depth.tiff
|   |   |-- ...
|   |-- 1a2a5a06ce083786581bb5a25b17bed6
|   |-- ...
|   |-- 1a3efcaaf8db9957a010c31b9816f48b
|   |-- ...
|-- JacquardV2_Dataset_1
|......
|-- JacquardV2_Dataset_2
|......
|-- JacquardV2_Dataset_3
|......
```

### Step 3

Edit your code carefully before running. 

Here we provide only a brief demonstration of the code. For detailed information, please refer to _**train.py**_.&#x20;

```
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Please add the downloaded Jacquard V2 directory into your python project. 
from jacquard_data import make_dataset, make_dataloader

dataset_root = '/data/JacquardV2_Dataset'  # path will not be same in your server.

train_dataset = make_dataset(dataset_root, config)
val_dataset = make_dataset(dataset_root, config)
train_loader = make_dataloader(train_dataset, is_training=True, **config['train_loader'])
val_loader = make_dataloader(val_dataset, is_training=False, **config['validation_loader'])

# Coding

```

### Step 4

Now you can start the implementation. For example, using the commands below:&#x20;

```
cd your_project_dir
// make coding
python your_script_name.py path_to_dataset_dir your_config.yaml
```

## Reference

Please cite the following paper if you find Jacquard V2 dataset and toolbox benefit your research. Thank you for your support!&#x20;
```
@inproceedings{li2024jacquardv2,
  title={Jacquard V2: Refining Datasets using the Human In the Loop Data Correction Method},
  author={Li, Qiuhao  and Yuan, Shenghai},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2024},
  organization={IEEE}
}
```
