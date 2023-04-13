# paper & code

reference paper: 

- [[1711.06025\] Learning to Compare: Relation Network for Few-Shot Learning (arxiv.org)](https://arxiv.org/abs/1711.06025)
- [Remote Sensing | Free Full-Text | Deep Relation Network for Hyperspectral Image Few-Shot Classification (mdpi.com)](https://www.mdpi.com/2072-4292/12/6/923)

reference code:  

- [floodsung/LearningToCompare_FSL: PyTorch code for CVPR 2018 paper: Learning to Compare: Relation Network for Few-Shot Learning (Few-Shot Learning part) (github.com)](https://github.com/floodsung/LearningToCompare_FSL)
-  [gokling1219/RN-FSC: Deep Relation Network for Hyperspectral Image Few-Shot Classification (github.com)](https://github.com/gokling1219/RN-FSC)
- [EnayatAria/ICA-based-band-selection-HSI: Independent component analysis for dimensionality reduction of hyperspectral images (github.com)](https://github.com/EnayatAria/ICA-based-band-selection-HSI)

# environment

## laptop windows 11

> **env**：Miniconda / Python 3.9.6 / Cuda 11.6
>
> **GPU**：NVIDIA GeForce GTX 1650 4GB
>
> **CPU**：lntel(R) Core(TM) i5-9300H CPU @ 2.40GHz
>
> **memory**：32GB

``` markdown
Python 3.9.16 (main, Mar  8 2023, 10:39:24) [MSC v.1916 64 bit (AMD64)] on win32

torch                             1.12.0+cu116
torchvision                       0.13.0+cu116
scikit-learn                      1.2.2
numpy                             1.24.2
visdom                            0.2.4

h5py                              3.8.0
scipy                             1.10.1
spectral                          0.23.1
mat73                             0.60

jupyter                           1.0.0
ipykernel                         6.22.0
ipython                           8.12.0
```

## server Ubuntu 20.04

[AutoDL-品质GPU租用平台-租GPU就上AutoDL](https://www.autodl.com/home)

> **Image **：PyTorch 1.11.0 / Python 3.8(ubuntu20.04) / Cuda 11.3
>
> **GPU**：RTX 3080(10GB) * 1
>
> **CPU**：12 vCPU Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz
>
> **memory**：40GB

> **Image **： PyTorch 1.11.0 / Python 3.8(ubuntu20.04) / Cuda 11.3
>
> **GPU**：RTX 3090(24GB) * 1
>
> **CPU**:：24 vCPU AMD EPYC 7642 48-Core Processor
>
> **memory**:：80GB



# directory

``` markdown
(dlpy39pth112) PS D:\Document\DevelopProject\Develop_DeepLearning\HSI\HSI-FSC> tree /f
Folder PATH listing for volume DATA
Volume serial number is 100A-93DC
D:.
│  .gitignore
│  LICENSE
│  README.md
│  requirements.txt
│  记录.md
│  记录.pdf
├─dataset
├─h5dataset_ica_bandselect_200
├─HSI_FSC ⭐⭐⭐
│  │  EM_RN_model.py
│  │  fewshot_train.py
│  │  generate_meta_dataset.py
│  │  generate_source_dataset.py
│  │  generate_target_dataset.py
│  │  Hyperparameters.md
│  │  meta_train_EM_RN.py
│  │  test.py
│  │  utils.py
│  ├─log
│  ├─model
│  ├─result
├─ICA-based-band-selection-HSI
│      bandselect_name_bands_sorted.py
│      ICA-based_for_BS_all.py
├─LearningToCompare_FSL-master
├─resultDataProcess
├─RN-FSC-main
├─RN_FSC_modify
├─server
└─Try_tmp

```



# run

## 1. visdom

### local

``` bash
python -m visdom.server
```

### server

``` bash
# 步骤一：在服务器上install visdom的0.1.8.9的版本 
pip install visdom==0.1.8.9 

# 步骤二：服务器上启动visdom 
python -m visdom.server

# 步骤三：本地ssh连接服务器并映射端口
ssh -L <本地端口>:localhost:8097 -p <ssh访问服务器的端口> <服务器用户名>@<ssh访问服务器的ip>
# 本地端口号可以随便设置，服务器用户名和ssh访问服务器的ip都可以在AutoDL中查看到
ssh -L 8080:localhost:8097  -p 23844 root@region-11.autodl.com 
#或
ssh -L 8080:container-e19b1182ac-a63c0c57:8097  -p 23844 root@region-11.autodl.com 

# 步骤四：全部完成，在本地网页中输入" localhost:8080 "

> [服务器visdom的本地显示_autodl vis_江南綿雨的博客-CSDN博客](https://blog.csdn.net/weixin_43702653/article/details/127273564)
```

## 2. band select

``` bash
cd ICA-based-band-selection-HSI
```

``` bash
# band select
python ICA-based_for_BS_all.py

# selected bands sorted
python bandselect_name_bands_sorted.py
```

## 3. generate dataset
### 3.1 generate source dataset

``` bash
cd HSI_FSC
```

``` bash
python generate_source_dataset.py --datasetname HS
python generate_source_dataset.py --datasetname BO
python generate_source_dataset.py --datasetname KSC
python generate_source_dataset.py --datasetname CH

python generate_meta_dataset.py
```

### 3.2 generate target dataset
``` bash
python generate_target_dataset --dataset XX

# XX: SA/IP/UP/PC/XZ
```

## 4. meta train

``` bash
python meta_train_EM_RN.py
```

## 5. fewshot train

``` bash
python fewshot_train.py --datasetname XX
```

## 6. test

``` bash
python test.py --datasetname XX
```

# adjust the parameters


RN_TargetDection_1_Learningrate

RN_TargetDection_2_dropout

RN_TargetDection_3_BactchNorm

RN_TargetDection_4_c_way

RN_TargetDection_5_k_shot_n_query

RN_TargetDection_6_inception

RN_TargetDection_7_networkstructure

RN_TargetDection_finnal

optimizer


# git push error!

``` bash
# 设置代理
(base) PS D:\Document\DevelopProject\Develop_DeepLearning\HSI\HSI-FSC> git config --global http.proxy http://127.0.0.1:7890
(base) PS D:\Document\DevelopProject\Develop_DeepLearning\HSI\HSI-FSC> git config --global https.proxy http://127.0.0.1:7890
(base) PS D:\Document\DevelopProject\Develop_DeepLearning\HSI\HSI-FSC> git config --global https.proxy
http://127.0.0.1:7890
(base) PS D:\Document\DevelopProject\Develop_DeepLearning\HSI\HSI-FSC> git config --global http.proxy
http://127.0.0.1:7890
# 取消代理
git config --global --unset http.proxy 
git config --global --unset https.proxy
```