# NSSNN

The implementation of TGRS 2022 paper ["Nonlocal Spatial-Spectral Neural Network for Hyperspectral Image Denoising"](https://ieeexplore.ieee.org/abstract/document/9930129/)

## Requisites
* See ```torch_37.yaml```

## Quick Start

### 1. Preparing your training/testing datasets

* Download HSIs from [here](https://njusteducn-my.sharepoint.com/:f:/g/personal/119106032867_njust_edu_cn/EhlvptVmZohEpjkNnu9P_xQBCJfpcSzXTg_omD2YCvXuIA?e=Xzy29C).(uploading)

#### Training dataset

* Create training datasets by ```python utility/lmdb_data.py```

#### Testing dataset

*Note matlab is required to execute the following instructions.*

* You can use the testing set we prepared for you in ```datasets/test/```

* Read the matlab code of ```matlab/generate_dataset*``` to understand how we generate noisy HSIs.

* Read and modify the matlab code of ```matlab/HSIData.m``` to generate your own testing dataset

### 2. Testing with pretrained models

* Our pretrained models are in ```checkpoints/```, you can use the scripts ```eval*.sh``` to test the pretrained models.

### 3. Training from scratch

* Use training scipts ```train*.sh``` to train your own models.

## Citation
If you find this work useful for your research, please cite: 
```
@ARTICLE{fu2022nssnn,
  author={Fu, Guanyiman and Xiong, Fengchao and Lu, Jianfeng and Zhou, Jun and Qian, Yuntao},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Nonlocal Spatial–Spectral Neural Network for Hyperspectral Image Denoising}, 
  year={2022},
  volume={60},
  number={},
  pages={1-16},
  doi={10.1109/TGRS.2022.3217097}}

## Contact
Please contact me if there is any question (gym.fu@njust.edu.cn)  
