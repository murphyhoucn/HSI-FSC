import numpy as np
import h5py
from scipy.io import loadmat


def Patch(data,height_index,width_index,PATCH_SIZE):   # PATCH_SIZE为一个patch（边长-1）的一半    data维度(H,W,C)
    height_slice = slice(height_index-PATCH_SIZE, height_index+PATCH_SIZE)
    width_slice = slice(width_index-PATCH_SIZE, width_index+PATCH_SIZE)
    # 由height_index和width_index定位patch中心像素
    patch = data[height_slice, width_slice,:]
    patch = patch.reshape(-1,patch.shape[0]*patch.shape[1]*patch.shape[2])
    # print(patch.shape)
    return patch

import sys
seed_number = sys.argv[1]
np.random.seed(int(seed_number))

img = loadmat('../../dataset/13_Xuzhou/xuzhou.mat')['xuzhou']
gt = loadmat('../../dataset/13_Xuzhou/xuzhou_gt.mat')['xuzhou_gt']
print(img.shape)  #

'''
波段处理
'''
bands_select = [87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 140, 141, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 191, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 208, 209, 210, 211, 212, 213]
for i in range(len(bands_select)):
    bands_select[i] = bands_select[i] - 1 
# print(bands_select)

'''
波段选择
'''
img = img[:, :, bands_select]
print(img.shape)  #

'''
图像归一化
'''
img = ( img * 1.0 - img.min() ) / ( img.max() - img.min() )
type(img)



[m, n, b] = img.shape
label_num = gt.max()  
print(label_num) # 9
PATCH_SIZE = 14   #每一个patch边长大小为9

# padding the hyperspectral images
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

print(img.shape, gt.shape)
# (528, 288, 100) (528, 288)


[m, n, b] = img.shape
count = 0 #统计有多少个中心像素类别不为0的patch
data = []
label = []

for i in range(PATCH_SIZE, m - PATCH_SIZE):
    for j in range(PATCH_SIZE, n - PATCH_SIZE):
        if gt[i, j] == 0:
            continue
        else:
            count += 1
            temp_data = Patch(img, i, j, PATCH_SIZE)
            #temp_label = np.zeros((1, label_num), dtype=np.int8)  # temp_label为一行九列[0,1,2,....,7,8]表示类别
            #temp_label[0, gt[i, j] - 1] = 1
            temp_label = gt[i, j] - 1 # 直接用0-8表示，不用独热编码
            data.append(temp_data)  # 每一行表示一个patch
            label.append(temp_label)
            # gt_index = ((i - PATCH_SIZE) * 217 + j - PATCH_SIZE)  # 记录坐标，用于可视化分类预测结果
            # f.write(str(gt_index) + '\n')
            # f1.write(str(temp_label) + '\n')
print(count)  # 68877

sample_count = count

data = np.array(data)
print(data.shape)  # (68877, 1, 78400)

data = np.squeeze(data)
print("squeeze : ", data.shape)  # squeeze :  (68877, 78400)

label = np.array(label)
print(label.shape)  # (68877,)
label = np.squeeze(label)
print("squeeze : ", label.shape)  # queeze :  (68877,)
print(np.unique(label)) # [ 0  1  2  3  4  5  6  7  8]


f = h5py.File('../../h5dataset_ica_bandselect_200/XZ_' + str(PATCH_SIZE*2) + '_' + str(PATCH_SIZE*2) + '_100_test.h5', 'w')
f['data'] = data
f['label'] = label
f.close()
print("XUZHOU test success")
# 每类随机采样num_s个生成支撑样本集

# 参数
num_s = 5  # 支撑样本集数量

indices = np.arange(data.shape[0]) 
shuffled_indices = np.random.permutation(indices)
data = data[shuffled_indices]
label = label[shuffled_indices]
data_s = []
label_s = []

for i in range(label_num): # 类别顺换 0123...16
    count = 0
    for j in range(sample_count): # 数量循环 : PC
        if label[j] == i and count <= num_s-1: # 如果标记为第i类
            data_s.append(data[j, :])
            label_s.append(label[j])
            count += 1
    print(count)

# 5
# 5
# 5
# 5
# 5
# 5
# 5
# 5
# 5

data_s = np.array(data_s)
label_s = np.array(label_s)

print(data_s.shape)
print(np.unique(label_s))
print(label_s.shape)

# (45, 78400)   # 45 = 9 * 5
# [ 0  1  2  3  4  5  6  7  8]
# (45,)

PATH = '../../h5dataset_ica_bandselect_200/XZ_' + str(PATCH_SIZE*2) + '_' + str(PATCH_SIZE*2) + '_100_support' + str(num_s) + '.h5'
f = h5py.File(PATH, 'w')
f['data_s'] = data_s
f['label_s'] = label_s
f.close()
print("SUZHOU SUPPORT success")