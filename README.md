# Joint Domain Adaptation with Weight Self-Learning for Hyperspectral Few-Shot Classification

This repo provides a Matlab implementation of paper “Joint Domain Adaptation with Weight Self-Learning for Hyperspectral Few-Shot Classification,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (IEEE JSTARS), DOI: 10.1109/JSTARS.2025.3601388.

This code is ONLY released for academic use. Please do not further distribute the code (including the download link), or put any of the code on the public website.

Please kindly cite our paper if you use our code in your research. Thanks and hope you will benefit from our code.

@ARTICLE{11134056, author={Kong, Lingyu and Sun, Xudong and Zhang, Jiahua and Wang, Xiaopeng and Shang, Xiaodi}, journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, title={JDAWSL: Joint Domain Adaptation With Weight Self-Learning for Hyperspectral Few-Shot Classification}, year={2025}, volume={18}, number={}, pages={21476-21493}, keywords={Feature extraction;Hyperspectral imaging;Adaptation models;Data models;Training;Metalearning;Few shot learning;Data augmentation;Accuracy;Three-dimensional displays;Adaptive learner;cross-domain learning;few-shot classification;hyperspectral imagery;joint domain adaptation (DA)}, doi={10.1109/JSTARS.2025.3601388}}

## dataset

You can download the preprocessed source domain data set (Chikusei_imdb_128.pickle) directly in pickle format, which is available in "https://pan.baidu.com/s/1s17xLfJr_CksGXqOlf2PZw?pwd=o8xu" , and move the files to `./datasets` folder.

The QUH-Qingyun and QUH-Pingan datasets are available at: https://github.com/Hang-Fu/QUH-classification-dataset.

IP and SA datasets are available at: http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes.

## example

An example dataset folder has the following structure:
```
datasets
├── Chikusei_imdb_128.pickle
├── IP
│   ├── indian_pines_corrected.mat
│   ├── indian_pines_gt.mat
├── SA
│   ├── ...
│   └── ...
├── QY
│   ├── ...
│   └── ...
└── Pa
    ├── ...
    └── ...
```

Meanwhile, for the convenience of saving the experimental results, we saved the model weights and prediction files to the `checkpoints` and `classificationMap` folders, the specific file structure is shown below.

```
checkpoints
├── IP
│   └── model weight files...
├── SA
│   └── ...
├── QY
│   └──...
└── Pa
    └── ...
    
classificationMap
├── IP
│   └── prediction mat file
├── SA
│   └── ...
├── QY
│   └── ...
└── Pa
    └── ...
```
