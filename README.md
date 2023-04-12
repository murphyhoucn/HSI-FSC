# 1. visdom

## 本地

``` bash
python -m visdom.server
```

## 云服务器

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
```

# 2. band select

``` bash
cd ICA-based-band-selection-HSI

# band select
python ICA-based_for_BS_all.py

# selected bands sorted
python bandselect_name_bands_sorted.py
```

# 3. generate dataset
## 3.1 generate source dataset

``` bash
cd source_dataset_process
jupyter notebook
# 使用jupyter notebook，运行四个source dataset的处理程序和一个将处理生成的文件合并的程序。
```

## 3.2 generate target dataset
``` bash
python generate_target_dataset --dataset XX
```

# 4. meta train

``` bash
python meta_train_EM_RN.py
```

# 5. fewshot_train_RN

``` bash
python fewshot_train.py --datasetname XX
```

# 6. test

``` bash
python test.py --datasetname XX
```



RN_TargetDection_1_Learningrate

RN_TargetDection_2_dropout

RN_TargetDection_3_BactchNorm

RN_TargetDection_4_c_way

RN_TargetDection_5_k_shot_n_query

RN_TargetDection_6_inception

RN_TargetDection_7_networkstructure

RN_TargetDection_finnal

optimizer


``` bash
# 设置代理
(base) PS D:\Document\DevelopProject\Develop_DeepLearning\HSI\HSI-FSC>
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