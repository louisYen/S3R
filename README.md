# Self-Supervised Sparse Representation for Video Anomaly Detection

![](https://i.imgur.com/bezMeKt.png =25x)
By Jhih-Ciang Wu, He-Yen Hsieh, Ding-Jie Chen, Chiou-Shann Fuh, and Tyng-Luh Liu


This repo is the official implementation of "**Self-Supervised Sparse Representation for Video Anomaly Detection**" (accepted at ECCV'22) for the weakly-supervised VAD (wVAD) setting.

<table width="100%" border=1 frame=void rules=cols>
  <tr>
  <td style="border-left-style:none; border-right-style:none;">
    <b>Table of Contents</b><br><br>
    <a href="#0">0. Introduction</a><br>
    <a href="#1">1. Quick start</a><br>
    <a href="#2">2. Prerequisitesn</a><br>
    <a href="#3">3. Installation</a><br>
    <a href="#4">4. Data preparation</a><br>
    <a href="#5">5. Dictionary learning</a><br>
    <a href="#6">6. Results and Models</a><br>
    <a href="#7">7. Evaluation</a><br>
    <a href="#8">8. Training</a><br>
    <a href="#9">9. Acknowledgement</a><br>
    <a href="#10">10. Citation</a><br>
  </tr>
</table>


![](https://i.imgur.com/w5vt2Sx.png)

## <a name="0"></a> Introduction

We consider establishing a dictionary learning approach to model the concept of *anomaly* at the feature level. The dictionary learning presumes an overcomplete basis, and prefers a sparse representation to succinctly explain a given sample. With the training set $\mathcal{X}$, whose video samples are anomaly-free, we are motivated to learn its corresponding dictionary $D$ of $N$ atoms. Since the derivation of $D$ is specific to the training dataset $\mathcal{X}$, we will use the notation $D_T$ to emphasize that the underlying dictionary is *task-specific*. With the learned task-specific dictionary $D_T$, we can design two opposite network components: the ***en-Normal*** and ***de-Normal*** modules. Given a snippet-level feature $F$, the former is used to obtain its reconstructed normal-event feature, while, on the contrary, the latter is applied to filter out the normal-event feature. The two modules complement each other and are central to our approach to anomaly video detection.

## <a name="1"></a> Quick start
```bash=
# please refer to the "Installation" section
$ conda create --name s3r python=3.6 -y
$ conda activate s3r
$ conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
$ cd S3R/
$ pip install -r requirements.txt

# please refer to the "Data preparation" section
$ ln -sT <your-data-path>/SH_Train_ten_crop_i3d data/shanghaitech/i3d/train
$ ln -sT <your-data-path>/SH_Test_ten_crop_i3d data/shanghaitech/i3d/test

# please refer to the "Dictionary learning" section
$ ln -sT <downloaded-dictionary-path>/ dictionary

# please refer to the "Evaluation" section
$ CUDA_VISIBLE_DEVICES=0 python tools/trainval_anomaly_detector.py \
--dataset shanghaitech --inference --resume checkpoint/shanghaitech_s3r_i3d_best.pth
```


## <a name="2"></a> Prerequisites
- <a href="https://www.linux.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/linux/linux-original.svg" alt="linux" width="30" height="30"/> </a> Operating system
    - Ubuntu 18.04.6 LTS
- <a href="https://developer.nvidia.com/cuda-toolkit" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/sco/2/21/Nvidia_logo.svg" alt="pytorch" width="30" height="30"/> </a> Graphics card
    - GPU: NVIDIA RTX 2080 Ti
- <a href="https://pytorch.org/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg" alt="pytorch" width="30" height="30"/> </a> Framework and environment
    - pytorch: 1.6.0
    - cuda: 10.1
    - torchvision: 0.7.0
- <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="30" height="30"/> </a> Programming language
    - python: 3.6

### Library versions for reference

The following information denotes the versions of installed libraries in our experiments.

- <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/3/39/Book.svg" alt="python" width="30" height="30"/> </a> Library versions
    - pyyaml==6.0
    - tqdm==4.64.0
    - munch==2.5.0
    - terminaltables==3.1.0
    - scikit-learn==0.24.2
    - opencv-python==4.6.0
    - pandas==1.1.5
    - typed-argument-parser==1.7.2
    - einops==0.4.1

### Project structure
```shell
$ tree S3R
S3R/
├─ anomaly/    (directory for core functions, including dataloader, S3R modules, and other useful functions)
├─ checkpoint/ (directory for model weights)
├─ configs/    (directory for model configurations)
├─ data/       (directory for dataset)
├─ dictionary/ (directory for learned dictionaries)
├─ tools/      (directory for main scripts)
├─ logs/       (directory for saving training logs)
├─ output/     (directory for saving inference results)
├─ config.py
├─ README.md 
├─ requirements.txt 
├─ utils.py
```   

## <a name="3"></a> Installation

**Step 1.** Create a conda environment and activate it.
```shell=
$ conda create --name s3r python=3.6 -y
$ conda activate s3r
```

**Step 2.** Install pytorch
```shell=
$ conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
or
$ pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

**Step 3.** Install required libraries
```shell=
$ pip install -r requirements.txt
```


## <a name="4"></a> Data preparation

Please download the extracted I3d features for shanghaitech and ucf-crime dataset from the [link](https://github.com/tianyu0207/RTFM).

The file structure of downloaded features should look like:
```bash
$ tree data
data/
├─ shanghaitech/
│  ├─ shanghaitech.training.csv
│  ├─ shanghaitech_ground_truth.testing.json
│  ├─ shanghaitech.testing.csv
│  ├─ i3d/
│  │  ├─ test/
│  │  │  ├─01_0015_i3d.npy
│  │  │  ├─05_033_i3d.npy
│  │  │  ├─ ...
│  │  ├─ train/
│  │  │  ├─ 01_0014_i3d.npy
│  │  │  ├─ 05_040_i3d.npy
│  │  │  ├─ ...
├─ ucf-crime/
│  ├─ ucf-crime_ground_truth.testing.json
│  ├─ ucf-crime.testing.csv
│  ├─ ucf-crime.training.csv
│  ├─ i3d/
│  │  ├─ test/
│  │  │  ├─ Abuse028_x264_i3d.npy
│  │  │  ├─ Burglary079_x264_i3d.npy
│  │  │  ├─ ...
│  │  ├─ train/
│  │  │  ├─ Abuse001_x264_i3d.npy
│  │  │  ├─ Burglary001_x264_i3d.npy
│  │  │  ├─ ...
```


Examples:

```bash=
$ ln -sT <your-data-path>/SH_Train_ten_crop_i3d data/shanghaitech/i3d/train
$ ln -sT <your-data-path>/SH_Test_ten_crop_i3d data/shanghaitech/i3d/test
$ ln -sT <your-data-path>/UCF_Train_ten_crop_i3d data/ucf-crime/i3d/train
$ ln -sT <your-data-path>/UCF_Test_ten_crop_i3d data/ucf-crime/i3d/test
```
## <a name="5"></a> Dictionary learning
The dictionaries can be downloaded from the [link](https://drive.google.com/drive/folders/1roEdnWUyCPQeur84I1X08JZSgZCYuQKK?usp=sharing) and the file structure of dictionaries should look like:
```bash
$ tree dictionary
dictionary/
├─ kinetics400
│  ├─ kinetics400_dictionaries.universal.omp.100iters.npy
├─ shanghaitech
│  ├─ shanghaitech_dictionaries.taskaware.omp.100iters.90pct.npy
│  ├─ shanghaitech_regular_features-2048dim.training.pickle
├─ ucf-crime
│  ├─ ucf-crime_dictionaries.taskaware.omp.100iters.50pct.npy
│  ├─ ucf-crime_regular_features-2048dim.training.pickle
```

Example:
```bash=
$ ln -sT <downloaded-dictionary-path>/ dictionary
```

### Generate dictionaries
To generate dictionaries for the shanghaitech and ucf-crime dataset, please run the following commands:

```bash=
# for the shanghaitech dataset
$ python data/shanghaitech/shanghaitech_dictionary_learning.py
and
# for the ucf-crime dataset
$ python data/ucf-crime/ucf_crime_dictionary_learning.py
```


## <a name="6"></a> Results and Models

|      config     |    dataset   | backbone | gpus | AUC (%) |                                             ckpt                                            |                                            log                                            |
|:---------------:|:------------:|:--------:|:----:|:-------:|:-------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------:|
| [shanghaitech_dl](/configs/shanghaitech/shanghaitech_dl.py) | shanghaitech |    [I3D](https://github.com/Tushar-N/pytorch-resnet3d)   |   1  |  97.40  | [model](https://drive.google.com/file/d/1f4KaRtwDzq3h9vR--nQRbRpXsraEkJ97/view?usp=sharing) | [log](/logs/shanghaitech_s3r_i3d.score) |
|   [ucf_crime_dl](/configs/ucf_crime/ucf_crime_dl.py)  |   ucf-crime  |    [I3D](https://github.com/Tushar-N/pytorch-resnet3d)   |   1  |  85.99  | [model](https://drive.google.com/file/d/1b6_WSkJAsaDQDVJJlxL2PxvkmBd7Gc9p/view?usp=sharing) | [log](/logs/ucf-crime_s3r_i3d.score) |




## <a name="7"></a> Evaluation
To evaluate the S3R on **shanghaitech**, please run the following command:
```bash=
$ CUDA_VISIBLE_DEVICES=0 python tools/trainval_anomaly_detector.py \
--dataset shanghaitech --inference --resume checkpoint/shanghaitech_s3r_i3d_best.pth

+ Performance on shanghaitech ----+---------+
|   Dataset    | Method | Feature | AUC (%) |
+--------------+--------+---------+---------+
| shanghaitech |  S3R   |   I3D   |  97.395 |
+--------------+--------+---------+---------+
```

To evaluate the S3R on **ucf-crime**, please run the following command:
```bash=
$ CUDA_VISIBLE_DEVICES=0 python tools/trainval_anomaly_detector.py \
--dataset ucf-crime --inference --resume checkpoint/ucf-crime_s3r_i3d_best.pth

+ Performance on ucf-crime ----+---------+
|  Dataset  | Method | Feature | AUC (%) |
+-----------+--------+---------+---------+
| ucf-crime |  S3R   |   I3D   |  85.989 |
+-----------+--------+---------+---------+
```


## <a name="8"></a> Training

### shaghaitech dataset
To train the S3R from scratch on **shanghaitech**, please run the following command:
```bash=
$ CUDA_VISIBLE_DEVICES=<gpu-id> python tools/trainval_anomaly_detector.py \
--dataset shanghaitech --version <customized-version> --evaluate_min_step 5000
```

Example:
```bash=
$ CUDA_VISIBLE_DEVICES=0 python tools/trainval_anomaly_detector.py \
--dataset shanghaitech --version s3r-vad-0.1 --evaluate_min_step 5000
```

### ucf-crime dataset
To train the S3R from scratch on **ucf-crime**, please run the following command:
```bash=
$ CUDA_VISIBLE_DEVICES=<gpu-id> python tools/trainval_anomaly_detector.py \
--dataset ucf-crime --version <customized-version> --evaluate_min_step 10
```

Example:
```bash=
$ CUDA_VISIBLE_DEVICES=0 python tools/trainval_anomaly_detector.py \
--dataset ucf-crime --version s3r-vad-0.1 --evaluate_min_step 10
```


## <a name="9"></a> Acknowledgement
Our codebase is built based on [RTFM](https://github.com/tianyu0207/RTFM). We really appreciate the authors for the nicely organized code!

## <a name="10"></a> Citation
We hope the codebase is beneficial to you. If this repo works positively for your research, please consider citing our paper. Thank you for your time and consideration.
```
@inproceedings{WuHCFL22,
  author    = {Jhih-Ciang Wu and
               He-Yen Hsieh and
               Ding-Jie Chen and
               Chiou-Shann Fuh and
               Tyng-Luh Liu},
  title     = {Self-Supervised Sparse Representation for Video Anomaly Detection},
  booktitle = {ECCV},
  year      = {2022},
}
```
