import argparse
import sys
import numpy as np
import h5py
import time
from utils import (
    make_print_to_file, 
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


    ## save source dataset
    source_dataset_save_sample_per_class = 200
    save_source_dataset(DATASETNEME, data, label, band_select_method[0], source_dataset_save_sample_per_class, hs_dict, label_num)


def save_source_dataset(DATASETNEME, data, label, band_select_method, save_sample, hs_dict, label_num):
    path1 = \
    '../h5dataset_' + \
    band_select_method + '_bandselect_' + \
    str(save_sample) + '/'  # 200对于target dataset无特殊意义，对于source dataset，200指的是每类只保存了200个训练样本

    path2 = \
    DATASETNEME + '_' + \
    str(PATCH_SIZE*2) + '_' + \
    str(PATCH_SIZE*2) + \
    '_100' + \
    '.h5'

    indices = np.arange(data.shape[0])  # list 
    shuffled_indices = np.random.permutation(indices) # ndarray
    #### 这两步骤会花费巨量的时间 ####
    print()
    print("\033[0;33;40m{}\033[0m".format("it will spend a long time!"))
    start_time = time.time()
    images = data[shuffled_indices]
    labels = label[shuffled_indices]  # 打乱顺序
    print("\033[0;33;40m{} spend time : {}s \033[0m".format("it will spend a long time!", time.time() - start_time))
    print()

    y = labels
    n_classes = y.max() + 1
    t_labeled = []    

    ## 样本数低于200的类别直接被丢掉
    ## 样本数超过200的类别，在其中随机取200个（随机的操作由上面的shuffled_indices完成）
    check_list = []
    for k, v in hs_dict.items():
        if v > save_sample:
            check_list.append(k)
    print("\033[0;33;40m{} dataset has {} classes \033[0m".format(DATASETNEME, label_num))
    print("\033[0;33;40m{} dataset only {} classes samples more than {} \033[0m".format(DATASETNEME, len(check_list), save_sample))
    print("\033[0;33;40mthey are{}\033[0m".format(check_list))
    print()
    
    for c in range(n_classes): 
        if hs_dict[c] < save_sample:
            pass
        else:
            i = indices[y == c][:save_sample]
            t_labeled += list(i)  # 列表中元素增加

    t_images = images[t_labeled]
    print('t_images', t_images.shape)
    t_labels = labels[t_labeled]
    print('t_labels', t_labels.shape)

    print()
    print("\033[0;33;40m{} h5 {} dataset saving........................\033[0m".format(DATASETNEME, 'source'))
    print(path1 + path2)
    f = h5py.File(path1 + path2, 'w')
    f['data'] = t_images
    f['label'] = t_labels
    f.close()
    print("\033[0;33;40m{} h5 {} dataset save success!\033[0m".format(DATASETNEME, 'source'))
    print()
    
if __name__ == '__main__':
    make_print_to_file(path='./log', current_filename=sys.argv[0])
    main()