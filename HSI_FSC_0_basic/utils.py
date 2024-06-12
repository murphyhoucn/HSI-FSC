import sys
import os
import sys
import time

import numpy as np
import h5py
from scipy.io import loadmat

def dataset_nway(name):

    # 经过波段选择和每类只取200个样本点的操作后，生成的test.h5数据的基本信息
    # n = 54129  # SA的测试标签数量 16类标签
    # n = 42776 # UP  9类标签
    # n = 68877 # XZ   9类标签
    # n = 10249 # IP 16类标签
    # n = 148152 # PC  9类标签

    n = 0
    m = 0
    if name == 'SA':
        n = 16
        m = 54129
    elif name == 'UP':
        n = 9
        m = 42776
    elif name == 'XZ':
        n = 9
        m = 68877
    elif name == 'IP':
        n = 16
        m = 10249
    elif name == 'PC':
        n = 9
        m = 148152
    else:
        n = 0
        m = 0

    return n, m

def make_print_to_file(path='./', current_filename='default'):
    '''
    path  it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''
    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.path= os.path.join(path, filename)
            self.log = open(self.path, "a", encoding='utf8',)
            print("save:", os.path.join(self.path, filename))
 
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
 
        def flush(self):
            pass

    nowtime = time.localtime()
    nowtime_style = time.strftime("%Y-%m-%d_%H_%M_%S", nowtime)

    fileName = current_filename + '_' + nowtime_style
    sys.stdout = Logger(fileName + '.log', path=path + '/')
 
    #############################################################
    # 这里输出之后的所有的输出的print 内容即将写入日志
    #############################################################
    print(fileName.center(60,'*'))

    # ————————————————
    # 版权声明：本文为CSDN博主「JY丫丫」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
    # 原文链接：https://blog.csdn.net/xu380393916/article/details/100887083



def Patch(data,height_index,width_index,PATCH_SIZE):   # PATCH_SIZE为一个patch（边长-1）的一半    data维度(H,W,C)
    height_slice = slice(height_index-PATCH_SIZE, height_index+PATCH_SIZE)
    width_slice = slice(width_index-PATCH_SIZE, width_index+PATCH_SIZE)
    # 由height_index和width_index定位patch中心像素
    patch = data[height_slice, width_slice,:]
    patch = patch.reshape(-1,patch.shape[0]*patch.shape[1]*patch.shape[2])
    # print(patch.shape)
    return patch


def  get_dataset_path(dataset_name):
    img_path = 'none'
    gt_path = 'none'
    img_name = 'none'
    gt_name = 'none'

    # target dataset
    if dataset_name == 'SA':
        img_path = '../dataset/7_Salinas/Salinas_corrected.mat'
        gt_path = '../dataset/7_Salinas/Salinas_gt.mat'
        img_name = 'salinas_corrected'
        gt_name = 'salinas_gt'
    elif dataset_name == 'UP':
        img_path = '../dataset/5_University_of_Pavia/PaviaU.mat'
        gt_path = '../dataset/5_University_of_Pavia/PaviaU_gt.mat'
        img_name = 'paviaU'
        gt_name = 'paviaU_gt'
    elif dataset_name == 'XZ':
        img_path = '../dataset/13_Xuzhou/xuzhou.mat'
        gt_path = '../dataset/13_Xuzhou/xuzhou_gt.mat'
        img_name = 'xuzhou'
        gt_name = 'xuzhou_gt'
    elif dataset_name == 'IP':
        img_path = '../dataset/8_Indian_Pines/Indian_pines_corrected.mat'
        gt_path = '../dataset/8_Indian_Pines/Indian_pines_gt.mat'
        img_name = 'indian_pines_corrected'
        gt_name = 'indian_pines_gt'
    elif dataset_name == 'PC':
        img_path = '../dataset/6_Pavia_Center/Pavia.mat'
        gt_path = '../dataset/6_Pavia_Center/Pavia_gt.mat'
        img_name = 'pavia'
        gt_name = 'pavia_gt'

        # source dataset
    elif dataset_name == 'BO':
        img_path = '../dataset/2_Botswana/Botswana.mat'
        gt_path = '../dataset/2_Botswana/Botswana_gt'
        img_name = 'Botswana'
        gt_name = 'Botswana_gt'
    elif dataset_name == 'CH':
        img_path = '../dataset/4_Chikusei/Chikusei_MATLAB/HyperspecVNIR_Chikusei_20140729.mat'
        gt_path = '../dataset/4_Chikusei/Chikusei_MATLAB/HyperspecVNIR_Chikusei_20140729_Ground_Truth.mat'
        img_name = 'chikusei'
        gt_name = 'GT'
    elif dataset_name == 'HS':
        img_path = '../dataset/1_Houston/Houston.mat'
        gt_path = '../dataset/1_Houston/Houston_gt.mat'
        img_name = 'Houston'
        gt_name = 'Houston_gt'
    elif dataset_name == 'KSC':
        img_path = '../dataset/3_Kennedy_Space_Center/KSC.mat'
        gt_path = '../dataset/3_Kennedy_Space_Center/KSC_gt.mat'
        img_name = 'KSC'
        gt_name = 'KSC_gt'

    else:
        print("\033[0;37;41m there is some error! wrong dataset name!! \033[0m")

    return img_path, gt_path, img_name, gt_name


def load_HSI(datasetname, img_path, gt_path, img_name, gt_name):
    if datasetname == 'CH':
        img = h5py.File(img_path)[img_name]
        gt = loadmat(gt_path)[gt_name][0][0][0]
        img = np.array(img).transpose((2, 1, 0))
    else:
        img = loadmat(img_path)[img_name]
        gt = loadmat(gt_path)[gt_name]
    print("".format(datasetname, img.shape))
    print("\033[0;33;40m{} HSI img shape = {}\033[0m".format(datasetname, img.shape))
    return img, gt

def band_select(dataset_name, band_select_method):
    # source dataset
    if dataset_name == 'SA':
        ica_band = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100, 122, 123, 124, 125, 126, 127, 128, 177, 181, 182, 183, 184, 185, 186, 187]
        grbs_band = [2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10 ,11 ,12 ,13 ,14 ,15 ,16 ,17 ,18 ,19 ,20 ,21 ,22 ,23 ,24 ,25 ,26 ,27 ,28 ,29 ,31 ,33 ,34 ,35 ,36 ,37, 38 ,39 ,40 ,41 ,42 ,43 ,44 ,45 ,46 ,47 ,48 ,49 ,50 ,51 ,52 ,53 ,54 ,55 ,56 ,57 ,58 ,59 ,60 ,61 ,62 ,63 ,64 ,65 ,66 ,67 ,68, 69 ,70 ,71 ,72 ,73 ,74 ,75 ,76 ,77 ,78 ,79 ,80 ,81 ,82 ,83 ,84 ,85 ,86 ,87 ,88 ,89 ,90 ,91 ,92 ,93 ,94 ,95 ,96 ,97 ,98 ,99, 100 ,126 ,139 ,204]
    elif dataset_name == 'UP':
        ica_band = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 102, 103]
        grbs_band = [2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10 ,11 ,12 ,13 ,14 ,15 ,16 ,17 ,18 ,19 ,20 ,21 ,23 ,24 ,25 ,26 ,27 ,28 ,29 ,30 ,31 ,32 ,33 ,34 ,35 ,36, 37 ,38 ,39 ,40 ,41 ,42 ,43 ,44 ,45 ,46 ,47 ,48 ,49 ,50 ,51 ,53 ,54 ,55 ,56 ,57 ,58 ,59 ,60 ,61 ,62 ,63 ,64 ,65 ,66 ,67 ,68, 69 ,70 ,71 ,72 ,73 ,74 ,75 ,76 ,77 ,78 ,79 ,80 ,81 ,82 ,83 ,84 ,85 ,86 ,87 ,88 ,89 ,90 ,91 ,92 ,93 ,94 ,95 ,96 ,97 ,98 ,99, 100 ,101 ,102 ,103]
    elif dataset_name == 'XZ':
        ica_band = [87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 140, 141, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 191, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 208, 209, 210, 211, 212, 213]
        grbs_band = []
    elif dataset_name == 'IP':
        ica_band = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140]
        grbs_band = []
    elif dataset_name == 'PC':
        ica_band = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102]
        grbs_band = [1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10 ,11 ,12 ,13 ,14 ,15 ,16 ,17 ,18 ,19 ,21 ,22 ,23 ,24 ,25 ,26 ,27 ,28 ,29 ,30 ,31 ,32 ,33 ,34 ,35, 36 ,37 ,38 ,39 ,40 ,41 ,42 ,43 ,44 ,45 ,46 ,47 ,48 ,49 ,50 ,51 ,52 ,53 ,54 ,55 ,56 ,57 ,58 ,60 ,61 ,62 ,63 ,64 ,65 ,66 ,67, 68 ,69 ,70 ,71 ,72 ,73 ,74 ,75 ,76 ,77 ,78 ,79 ,80 ,81 ,82 ,83 ,84 ,85 ,86 ,87 ,88 ,89 ,90 ,91 ,92 ,93 ,94 ,95 ,96 ,97 ,98, 99 ,100 ,101 ,102]
    
    # target dataset
    elif dataset_name == 'BO':
        ica_band = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103]
        grbs_band = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 88, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 137, 138, 139, 140, 141, 142, 143, 144, 145]
    elif dataset_name == 'CH':
        ica_band = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 36, 37, 38, 39, 40, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128]
        grbs_band = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 65, 66, 67, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128]    
    elif dataset_name == 'HS':
        ica_band = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 138, 139, 140, 141, 143, 144]
        grbs_band = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 77, 107, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 132, 133, 134, 135, 143, 144]
    elif dataset_name == 'KSC':
        ica_band = [1, 66, 67, 68, 72, 74, 75, 76, 77, 81, 82, 83, 84, 85, 86, 87, 91, 93, 94, 95, 96, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176]
        grbs_band = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 31, 32, 33, 35, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 95, 101, 120, 132, 143, 144, 145, 146, 147, 148, 149, 150, 151, 155, 167, 175, 176]   
    else:
        print("\033[0;37;41m there is some error! wrong dataset name!! \033[0m")

    if band_select_method == 'ica':
        selected_bands = ica_band
    elif band_select_method == 'grbs':
        selected_bands = grbs_band
    for i in range(len(selected_bands)):
        selected_bands[i] = selected_bands[i] - 1 
    # print(selected_bands)

    return selected_bands


def convert_to_color(x, palette):
    return convert_to_color_(x, palette=palette)

def convert_to_color_(arr_2d, palette=None):
    """Convert an array of labels to RGB color-encoded image.

    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)

    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format

    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        raise Exception("Unknown color palette")

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def display_predictions(pred, vis, gt=None, caption=""):
    if gt is None:
        vis.images([np.transpose(pred, (2, 0, 1))],
                    opts={'caption': caption})
    else:
        vis.images([np.transpose(pred, (2, 0, 1)),
                    np.transpose(gt, (2, 0, 1))],
                    nrow=2,
                    opts={'caption': caption})
        # ============== <utils display_predictions> murphy 13-apr-23 =================
        vis.images([np.transpose(pred, (2, 0, 1))],
                    opts={'caption': caption})
        vis.images([np.transpose(gt, (2, 0, 1))],
                    opts={'caption': caption})
        # ============== <utils display_predictions> murphy 13-apr-23 =================
