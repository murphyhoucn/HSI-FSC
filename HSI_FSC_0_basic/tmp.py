import argparse
import numpy as np
import h5py
import time
from utils import (
    Patch, 
    get_dataset_path, 
    load_HSI, 
    band_select,
)

parser = argparse.ArgumentParser(description="generate source dataset")
parser.add_argument("-dn", "--datasetname", type = str, default = 'BO')
args = parser.parse_args()

DATASETNEME = args.datasetname
PATCH_SIZE = 14 # not clear

def main():
    ## load date
    img_path, gt_path, img_name, gt_name = get_dataset_path(DATASETNEME)
    img, gt = load_HSI(DATASETNEME, img_path, gt_path, img_name, gt_name)

    ## band select
    band_select_method = ['ica', 'grbs']
    selected_bands = band_select(DATASETNEME, band_select_method[0])
    img = img[:, :, selected_bands]
    print("\033[0;33;40m{} HSI_band_selected img shape = {}\033[0m".format(DATASETNEME, img.shape))

    ## normalization
    img = ( img * 1.0 - img.min() ) / ( img.max() - img.min() )
    # print(type(img))

    [m, n, b] = img.shape
    label_num = gt.max()
    print("\033[0;33;40m{} HSI label num(excluding unlabeled) = {}\033[0m".format(DATASETNEME, label_num))

    ## padding the hyperspectral images
    img_temp = np.zeros((m + 2 * PATCH_SIZE, n + 2 * PATCH_SIZE, b), dtype=np.float32)
    img_temp[PATCH_SIZE:(m + PATCH_SIZE), PATCH_SIZE:(n + PATCH_SIZE), :] = img[:, :, :]

    for i in range(PATCH_SIZE):
        img_temp[i, :, :] = img_temp[2 * PATCH_SIZE - i, :, :]
        img_temp[m + PATCH_SIZE + i, :, :] = img_temp[m + PATCH_SIZE - i - 2, :, :]

    for i in range(PATCH_SIZE):
        img_temp[:, i, :] = img_temp[:, 2 * PATCH_SIZE - i, :]
        img_temp[:, n + PATCH_SIZE + i, :] = img_temp[:, n + PATCH_SIZE  - i - 2, :]

    img = img_temp
    del img_temp
    gt_temp = np.zeros((m + 2 * PATCH_SIZE, n + 2 * PATCH_SIZE), dtype=np.int8)
    gt_temp[PATCH_SIZE:(m + PATCH_SIZE), PATCH_SIZE:(n + PATCH_SIZE)] = gt[:, :]
    gt = gt_temp
    del gt_temp
    print("\033[0;33;40m{} HSI & gt padding shape = {}, {}\033[0m".format(DATASETNEME, img.shape, gt.shape))

    ## count label sample number
    [m, n, b] = img.shape
    data = []
    label = []
    count = 0 #统计有多少个中心像素类别不为0的patch

    for i in range(PATCH_SIZE, m - PATCH_SIZE):
        for j in range(PATCH_SIZE, n - PATCH_SIZE):
            if gt[i, j] == 0:
                continue
            else:
                count += 1
                temp_data = Patch(img, i, j, PATCH_SIZE)
                temp_label = gt[i, j] - 1  # 1,2,3... -> 0,1,2...
                data.append(temp_data)
                label.append(temp_label)
    print("\033[0;33;40m{} HSI(unpadding) has label num = {}\033[0m".format(DATASETNEME, count))
    # count = 54129  # SA的测试标签数量 16类标签
    # count = 42776 # UP  9类标签
    # count = 148152 # PC  9类标签
    # count = 10249 # IP 16类标签
    # count = 68877 # XZ   9类标签

    # count = 3248  # BOT 14
    # count = 15029 # HS 15
    # count = 77592 # CH 19
    # count = 5211 # KSC 13

    print()
    print("\033[0;33;40m{}\033[0m".format("list to ndarray start! it will spend a long time!"))
    start_time = time.time()
    ## listdata trans to ndarray
    data = np.array(data)
    # print(data.shape)  # (54129, 1, 78400) # example SA
    data = np.squeeze(data)
    # print("squeeze : ", data.shape)  # squeeze :  (54129, 78400)  # example SA
    label = np.array(label)
    # print(label.shape)  # (54129,)  # example SA
    label = np.squeeze(label)
    # print("squeeze : ", label.shape)  # queeze :  (54129,)  # example SA
    # print(np.unique(label)) # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]  # example SA
    print("\033[0;33;40m{} spend time : {}s \033[0m".format("list to ndarray done!", time.time() - start_time))
    print()

    ## 查看类别对应的标记样本数
    hs_dict = {}
    hs_dict = hs_dict.fromkeys([i for i in range(label_num)], 0)
    # print(hs_dict)
    for i in label:
        if i in hs_dict:
            hs_dict[i] += 1
    print("\033[0;33;40m{} HSI dataset class and labels : {} \033[0m".format(DATASETNEME, hs_dict))
    hs_label_sum = sum(hs_dict.values())
    print("\033[0;33;40m{} HSI(unpadding) has label num = {}\033[0m".format(DATASETNEME, hs_label_sum))
    # 这里的结果和上面的count一样

    
if __name__ == '__main__':
    main()