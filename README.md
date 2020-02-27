# Remote-Sensing-Image-Classification
Official implementation for [Remote Sensing Image Classification via Improved Cross-Entropy Loss and Transfer Learning Strategy Based on Deep Convolutional Neural Networks](https://ieeexplore.ieee.org/abstract/document/8844264), IEEE Geoscience and Remote Sensing Letters 2019

## Citation
Please cite our project if it is helpful for your research
```
A. Bahri, S. G. Majelan, S. Mohammadi, M. Noori and K. Mohammadi, "Remote Sensing Image Classification via Improved Cross-Entropy Loss and Transfer Learning Strategy Based on Deep Convolutional Neural Networks," in IEEE Geoscience and Remote Sensing Letters.

```
-
-
<p align="center">
    <img src="https://github.com/AliBahri94/Remote-Sensing-Image-Classification/blob/master/docs/d5a84f12-91af-4b3d-8096-2d4a5e641ecc-usa.png">
</p> 
<p align="center">
    21 class UC Merced land-use Dataset (RGB)
</p>
-
-
<p align="center">
    <img src="https://github.com/AliBahri94/Remote-Sensing-Image-Classification/blob/master/docs/08191186-61c0-4298-80f0-85825f8ba2b4-udsd.png">
</p> 
<p align="center">
    Our Architecture
</p>

## Dependencies
- 1 Nvidia GPU (4h training on Titan Xp)
- ``Python3``
- ``tensorflow 1.15``
- ``numpy 1.17.5``
- ``keras 2.2.5``

## Download original datasets
- AID (Aerial Image Dataset)
- Download (https://drive.google.com/file/d/1D8gvnEvzbyNlZHLLD3zqLGiUaxgqp0yN/view?usp=sharing)
- NWPU-RESISC45 (Northwestern Polytechnic University Remote Sensing Image Scene Classification 45)
- Download (https://drive.google.com/file/d/1eOMQ7zF19KRvjxZqVESMYzAd9X6gZfar/view?usp=sharing)
- UC Merced land-use
- Download (https://drive.google.com/file/d/1rzNVDsRn3JcVNnAYCI_eyd43ZoU_5vuq/view?usp=sharing)
- WHU-RS19
- Download (https://drive.google.com/file/d/1KuTwHU9Yumswrp9K1_FK0dlMN8QRjN-y/view?usp=sharing)

## Use ready datasets (splited to train and valid parts)
- AID (Aerial Image Dataset) (train: 50%, valid: 50%)
- Address (https://drive.google.com/drive/folders/10U9jzimYUtD9iGn9am3cricgEpbfNuJV?usp=sharing)
- AID (Aerial Image Dataset) (train: 70%, valid: 30%)
- Address (https://drive.google.com/drive/folders/11hTqDCVB-hoEDWMTMIeGbPyAytKDQ8IA?usp=sharing)
- NWPU-RESISC45 (Northwestern Polytechnic University Remote Sensing Image Scene Classification 45) (train: 20%, valid: 80%)
- Address (https://drive.google.com/drive/folders/1X2oTWq8hJ-1Miy1mJyM4SD3l7SuZgN95?usp=sharing)
- NWPU-RESISC45 (Northwestern Polytechnic University Remote Sensing Image Scene Classification 45) (train: 30%, valid: 70%)
- Address (https://drive.google.com/drive/folders/1XJinSCqe8mLUcmj4KzW9nWzM8zhnW_g4?usp=sharing)
- UC Merced land-use (train: 80%, valid: 20%)
- Address (https://drive.google.com/drive/folders/15JMhL7peTdO8DZhyheYrkKbCUuFabeGT?usp=sharing)

### Pretrained models
- Download trained model on AID dataset (train: 70% , valid: 30%) with accuracy score: 98.10 (https://drive.google.com/file/d/1-2sb1gBU9oYN4SF-iZ4Xab1mwVnmk0AD/view?usp=sharing)
- Download trained model on AID dataset (train: 50% , valid: 50%) with accuracy score: 97.08 (https://drive.google.com/file/d/1-1fHZODRLKUvRwlCBHLVCMo4e7E-32HX/view?usp=sharing)
- Download trained model on UC Merced land-use dataset (train: 80% , valid: 20%) with accuracy score: 99.52 (https://drive.google.com/file/d/1-20x38XGckZCNM-wsV7Gvpif4jaCVRQN/view?usp=sharing)
- Download trained model on NWPU-RESISC45 dataset (train: 20% , valid: 80%) with accuracy score: 93.56 (https://drive.google.com/file/d/1-Ey8NkAa0HksmSrw7opIB1oATGD5_jH5/view?usp=sharing)
- Download trained model on NWPU-RESISC45 dataset (train: 30% , valid: 70%) with accuracy score: 94.44 (https://drive.google.com/file/d/1hYcdtJHrviuDLGYwoodzJC9b23FnUj2q/view?usp=sharing)

## Project layout (recommended)
```
Remote_Sensing_Image_Classification/
├── checkpoint
├── data
│   ├── AID (train:70%, valid:30%)
│   ├── AID (train:50%, valid:50%)
│   ├── UCMerced (train:80%, valid:20%)
│   ├── NWPU-RESISC45 (train:30%, valid:70%)
│   └── NWPU-RESISC45 (train:20%, valid:80%)
├── docs
└── model_pretrained
    ├── NasNet_Mobile_New_Loss3.02-0.9810(AID_70_30).h5
    ├── NasNet_Mobile_New_Loss3.19-0.9708(AID_50_50).h5
    ├── NasNet_Mobile_New_Loss3.117-0.9952(UCMerced_80_20).h5
    ├── NasNet_Mobile_New_Loss3_Dore_3.06-0.9356(NWPU_20_80).h5
    └── NasNet_Mobile_New_Loss3_94.43(NWPU_30_70).h5

```
## Quick start to validate(using ready datasets)
1. Use ready dataset path
2. Download pretrained models and put into ``checkpoint/`` directory
3. Run ``python predict.py``
4. Results will be shown.
- Note: Configurations is in the config.py file.

## Start to validate (using original datasets)
1. Download original dataset and put into ``data/`` directory.
2. Unzip dataset
3. Run ``python divide_dataset.py`` to split dataset to train and valid folder
4. Download pretrained models and put into ``checkpoint/`` directory
5. Run ``python predict.py``
6. Results will be shown.
- Note: Configurations is in the config.py file.

## Quick start to Training (using ready datasets)
1. Use ready dataset path
2. Run ``python train.py``
3. All Models with be saved into ``checkpoint/`` direcory
- Note: Configurations is in the config.py file.

## Start to Training (using original datasets)
1. Download original dataset and put into ``data/`` directory
2. Unzip dataset
3. Run ``python divide_dataset.py`` to split dataset to train and valid folder
4. Run ``python train.py``
5. All Models with be saved into ``checkpoint/`` direcory
- Note: Configurations is in the config.py file.

## Quantitative and Qualitative results
\usepackage{graphics}
\scalebox{5}
<p align="center">
    <img src="https://github.com/AliBahri94/Remote-Sensing-Image-Classification/blob/master/docs/cfab54f8-de43-48df-aaf7-8e1b47484013-uaaaaaa.png">
</p> 
<p align="center">
    Bootstrap Chart for NWPU-RESISC45 Dataset
</p>
-
-
<p align="center">
    <img src="https://github.com/AliBahri94/Remote-Sensing-Image-Classification/blob/master/docs/t_1.PNG">
</p> 
<p align="center">
    OVERALL ACCURACY OF THE REFERENCE AND THE PROPOSED METHOD ON THE UC MERCED DATASET
</p>
-
-
<p align="center">
    <img src="https://github.com/AliBahri94/Remote-Sensing-Image-Classification/blob/master/docs/t_2.PNG">
</p> 
<p align="center">
    OVERALL ACCURACY OF THE REFERENCE AND THE PROPOSED METHOD ON THE AID DATASET (50% TRAINING, 50% TESTING)
</p>
-
-
<p align="center">
    <img src="https://github.com/AliBahri94/Remote-Sensing-Image-Classification/blob/master/docs/t_3.PNG">
</p> 
<p align="center">
    OVERALL ACCURACY OF THE REFERENCE AND THE PROPOSED METHOD ON THE AID DATASET (70% TRAINING, 30% TESTING)
</p>
-
-
<p align="center">
    <img src="https://github.com/AliBahri94/Remote-Sensing-Image-Classification/blob/master/docs/t_4.PNG">
</p> 
<p align="center">
    OVERALL ACCURACY OF THE REFERENCE AND THE PROPOSED METHOD ON THE NWPU-RESISC45 DATASET (20% TRAINING,80% TESTING)
</p>
-
-
<p align="center">
    <img src="https://github.com/AliBahri94/Remote-Sensing-Image-Classification/blob/master/docs/t_5.PNG">
</p> 
<p align="center">
    OVERALL ACCURACY OF THE REFERENCE AND THE PROPOSED METHOD ON THE NWPU-RESISC45 DATASET (30% TRAINING,70% TESTING)
</p>
