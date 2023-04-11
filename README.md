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



# 2. meta_train_RN.py

``` bash
python meta_train_RN.py
```

# 3. generate_data

``` bash
cd target_dataset_process
python generate_data_XX.py 1
```

# 4. fewshot_train_RN

``` bash
python fewshot_train_RN_FT_XX.py
```

# 5. test

``` bash
python test_RN_XX.py
```



> 重复3（使用不同的随机种子）， 4， 5
>
> 将test输出的结果做平均，得到最后的结果。



RN_TargetDection_1_Learningrate

RN_TargetDection_2_dropout

RN_TargetDection_3_BactchNorm

RN_TargetDection_4_c_way

RN_TargetDection_5_k_shot_n_query

RN_TargetDection_6_inception

RN_TargetDection_7_networkstructure

RN_TargetDection_finnal

optimizer

