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

parser = argparse.ArgumentParser(description="generate target dataset")
parser.add_argument("-dn", "--datasetname", type = str, default = 'SA')
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
    # n = 54129  # SA的测试标签数量 16类标签
    # n = 42776 # UP  9类标签
    # n = 148152 # PC  9类标签
    # n = 10249 # IP 16类标签
    # n = 68877 # XZ   9类标签

    print()
    print("\033[0;33;40m{}\033[0m".format("list to ndarray start! it will spend a long time!"))
    start_time = time.time()
    ## listdata trans to ndarray
    data = np.array(data)
    # print(data.shape)  # (54129, 1, 78400)
    data = np.squeeze(data)
    # print("squeeze : ", data.shape)  # squeeze :  (54129, 78400)
    label = np.array(label)
    # print(label.shape)  # (54129,)
    label = np.squeeze(label)
    # print("squeeze : ", label.shape)  # queeze :  (54129,)
    # print(np.unique(label)) # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]
    print("\033[0;33;40m{} spend time : {}s \033[0m".format("list to ndarray done!", time.time() - start_time))
    print()

    ## save test
    save_test(DATASETNEME, data, label, band_select_method[0], 'test')

    ## save support
    support_number = 5
    save_support(DATASETNEME, data, label, band_select_method[0], 'support', support_number, label_num, count)

def save_test(DATASETNEME, data, label, band_select_method, mode):
    path1 = \
    '../h5dataset_' + \
    band_select_method + '_bandselect_' + \
    '200/'  # 200对于target dataset无特殊意义，对于source dataset，200指的是每类只保存了200个训练样本

    path2 = \
    DATASETNEME + '_' + \
    str(PATCH_SIZE*2) + '_' + \
    str(PATCH_SIZE*2) + \
    '_100_' + \
    mode + \
    '.h5'
    
    print("\033[0;33;40m{} h5 {} dataset saving........................\033[0m".format(DATASETNEME, mode))
    print(path1 + path2)
    f = h5py.File(path1 + path2, 'w')
    f['data'] = data
    f['label'] = label
    f.close()
    print("\033[0;33;40m{} h5 {} dataset save success!\033[0m".format(DATASETNEME, mode))

def save_support(DATASETNEME, data, label, band_select_method, mode, support_number, label_num, count):
    dif_supp_data_num = 10
    for jishujun in range (1, dif_supp_data_num + 1):
        path1 = \
        '../h5dataset_' + \
        band_select_method + '_bandselect_' + \
        '200/'  # 200对于ta rget dataset无特殊意义，对于source dataset，200指的是每类只保存了200个训练样本

        path2 = \
        DATASETNEME + '_' + \
        str(PATCH_SIZE*2) + '_' + \
        str(PATCH_SIZE*2) + \
        '_100_' + \
        mode + \
        str(support_number) + '_' + \
        str(jishujun) + \
        '.h5'

        indices = np.arange(data.shape[0]) 
        shuffled_indices = np.random.permutation(indices)  # 随机排列序列。
        #### 这两步骤会花费巨量的时间 ####
        print()
        print("\033[0;33;40m{}\033[0m".format("it will spend a long time!"))
        start_time = time.time()
        data = data[shuffled_indices]
        label = label[shuffled_indices]
        print("\033[0;33;40m{} spend time : {}s \033[0m".format("it will spend a long time!", time.time() - start_time))
        print()
        data_s = []
        label_s = []

        for i in range(label_num): # 类别循环 0123...16
            # print('**************{}************'.format(i))
            one_class_sample_count = 0
            for j in range(count): # 数量循环 54129:SA
                # print('=============={}============'.format(j))
                if label[j] == i and one_class_sample_count <= support_number - 1: # 如果标记为第i类
                    data_s.append(data[j, :])
                    label_s.append(label[j])
                    one_class_sample_count += 1
                if one_class_sample_count == support_number:
                    break
            # print(one_class_sample_count)      

        data_s = np.array(data_s)
        label_s = np.array(label_s)

        # print(data_s.shape)
        # print(np.unique(label_s))
        # print(label_s.shape)
        # (80, 78400)   # 80 = 16 * 5
        # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]
        # (80,)

        print("\033[0;33;40m{} h5 {} dataset {} saving........................\033[0m".format(DATASETNEME, mode, jishujun))
        print(path1 + path2)
        f = h5py.File(path1 + path2, 'w')
        f['data_s'] = data_s
        f['label_s'] = label_s
        f.close()
        print("\033[0;33;40m{} h5 {} dataset {} save success! \033[0m".format(DATASETNEME, mode, jishujun))

if __name__ == '__main__':
    make_print_to_file(path='./log', current_filename=sys.argv[0])
    main()