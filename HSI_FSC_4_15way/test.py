import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import argparse
import h5py
import time
from sklearn import metrics
import csv
import os
import sys
import visdom
from scipy.io import savemat

from utils import dataset_nway
from utils import make_print_to_file
from EM_RN_model import Embedding
from EM_RN_model import RelationNetwork
from EM_RN_model import weights_init

#torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")

parser.add_argument("-s","--n_shot",type = int, default = 5)       # support set per class
#parser.add_argument("-b","--n_query",type = int, default = 19)       # query set per class

parser.add_argument("-me","--meta_episode",type = int, default= 10000)
parser.add_argument("-fe","--fewshot_episode",type = int, default= 1000)

parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)

parser.add_argument("-dn","--datasetname", type=str, default='0')
parser.add_argument("-n","--count", type=int, default=0)
args = parser.parse_args()


# Hyper Parameters

n_shot = args.n_shot
FEW_SHOT_EPISODE = args.fewshot_episode
h5datasetname = args.datasetname
LEARNING_RATE = args.learning_rate
GPU = args.gpu
jishujun = args.count

n_way, n_samples = dataset_nway(h5datasetname)
if n_way == 0:
    print("\033[0;33;40m{}\033[0m".format(str("wrong!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")))

print("\033[0;33;40m{}\033[0m".format(str("check it")))
print("\033[0;33;40mdataset: {}\033[0m".format(h5datasetname))
print("\033[0;33;40mn_way: {}\033[0m".format(n_way))
print("\033[0;33;40mn_samples: {}\033[0m".format(n_samples))

current_path = os.getcwd()
# print(current_path)
path_list = current_path.split("/") # linux
# path_list = current_path.split("\\") # windwos
# print(path_list)
cur_filename = path_list[-1]
# print(cur_filename)
viz = visdom.Visdom(env = cur_filename)
if not viz.check_connection:
    print("Visdom is not connected. Did you run 'python -m visdom.server' ?")
viz_tmp_name1 = 'test_' + h5datasetname +'_result'
viz_tmp_name2 = 'test_' + h5datasetname +'_acc'
viz.text(' ',win= viz_tmp_name1, opts=dict(title=viz_tmp_name1))

n_examples = 5  # support每类5个样本
im_width, im_height, depth = 28, 28, 100 # 输入的cube为固定值

feature_encoder = Embedding()
relation_network = RelationNetwork()

feature_encoder.cuda(GPU)
relation_network.cuda(GPU)

feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=100000, gamma=0.5)
relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
relation_network_scheduler = StepLR(relation_network_optim, step_size=100000, gamma=0.5)


# 载入元学习的模型
model_name = './model'
model_dir_file = os.listdir(model_name)
print(model_dir_file)
model_dir_file_meta = []
for i in model_dir_file:
    if i[0] == h5datasetname[0] and int(i[-5]) == jishujun:
        model_dir_file_meta.append(i)
# model_dir_file_meta = [i for i in model_dir_file if i[0] == h5datasetname[0] and int(i[-5]) == jishujun]
print(model_dir_file_meta)

feature_encoder_model_name = model_name + '/' + model_dir_file_meta[0]
relation_network_model_name = model_name + '/' + model_dir_file_meta[1]

print("\033[0;33;40m{}\033[0m".format(feature_encoder_model_name))
print("\033[0;33;40m{}\033[0m".format(relation_network_model_name))

feature_encoder.load_state_dict(torch.load(feature_encoder_model_name))
print("load fewshot feature encoder success")

relation_network.load_state_dict(torch.load(relation_network_model_name))
print("load fewshot relation network success")

feature_encoder.eval()
relation_network.eval()


def kappa(testData, k): #testData表示要计算的数据，k表示数据矩阵的是k*k的
    dataMat = np.mat(testData)
    s = dataMat.sum()

    print('==========混淆矩阵===========')
    
    print(dataMat)
    tmp_mat_ndarray = dataMat.A

    nowtime = time.localtime()
    nowtime_style = time.strftime("%Y-%m-%d__%H_%M_%S", nowtime)

    # print("_++++++++__============================")
    # print(type(dataMat))
    # print(dataMat.shape)
    # print(type(tmp_mat_ndarray))
    # print("_++++++++__============================")

    cm_name = \
        './result/' + \
        h5datasetname + '_' + \
        'cm_' + \
        nowtime_style + '.mat'
    
    savemat(cm_name, {str(cm_name) : tmp_mat_ndarray})

    P0 = 0.0
    for i in range(k):
        P0 += dataMat[i, i]*1.0
    xsum = np.sum(dataMat, axis=1)
    ysum = np.sum(dataMat, axis=0)
    #xsum是个k行1列的向量，ysum是个1行k列的向量
    #Pe = float(ysum * xsum) / float(s * 1.0) / float(s * 1.0)
    Pe = float(ysum * xsum) / float(s ** 2)
    print("\033[0;33;40mPe ={}\033[0m".format(Pe))

    P0 = float(P0/float(s*1.0))
    #print("P0 = ", P0)
    cohens_coefficient = float((P0-Pe)/(1-Pe))

    a = []
    a = dataMat.sum(axis=0)
    a = np.float32(a)
    a = np.array(a)
    a = np.squeeze(a)

    for i in range(k):
        #print(dataMat[i, i])
        a[i] = float(dataMat[i, i]*1.0)/float(a[i]*1.0)

    AA = a.mean()*100
    print("\033[0;33;40mAA = {}\033[0m".format(AA))
    
    return cohens_coefficient, AA, a*100


def rn_predict(support_images, test_images, num):

    support_tensor = torch.from_numpy(support_images)
    query_tensor = torch.from_numpy(test_images)

    # calculate features
    sample_features = feature_encoder(Variable(support_tensor).cuda(GPU))  # 数量*通道*高度*宽度
    #print( list(sample_features.size()) ) # [9, 32, 6, 3, 3]
    sample_features = sample_features.view(n_way, n_shot, list(sample_features.size())[-4],
                                           list(sample_features.size())[-3],
                                           list(sample_features.size())[-2], list(sample_features.size())[
                                               -1])  # view函数改变shape: 5way, 5shot, 64, 19, 19
    # sample_features = torch.sum(sample_features, 1).squeeze(1)  # 同类样本作和
    sample_features = torch.mean(sample_features, 1).squeeze(1)  # 同类样本取平均
    #print( list(sample_features.size()) ) # [9, 32, 6, 3, 3]
    batch_features = feature_encoder(Variable(query_tensor).cuda(GPU))  # 20x64*5*5
    #print(list(batch_features.size())) # [1000, 32, 6, 3, 3]

    ################################################################################################################
    sample_features = sample_features.view(n_way, list(sample_features.size())[1] * list(sample_features.size())[2],
                                           list(sample_features.size())[-2], list(sample_features.size())[-1])
    batch_features = batch_features.view(num,
                                         list(batch_features.size())[1] * list(batch_features.size())[2],
                                         list(batch_features.size())[-2], list(batch_features.size())[-1])
    #print(list(sample_features.size())) # [9, 192, 3, 3]
    #print(list(batch_features.size())) # [1000, 192, 3, 3]
    ################################################################################################################

    # calculate relations
    # 支撑样本和查询样本进行连接
    sample_features_ext = sample_features.repeat(num, 1, 1, 1, 1)  # # repeat函数沿着指定的维度重复tensor
    #print(list(sample_features_ext.size())) # [380, 20, 128, 5, 5]
    batch_features_ext = batch_features.repeat(n_way, 1, 1, 1, 1)
    batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
    #print(list(batch_features_ext.size())) # [380, 20, 128, 5, 5]

    relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2)
    # print(list(relation_pairs.size())) # [380, 20, 256, 5, 5]
    relation_pairs = relation_pairs.view(-1, list(relation_pairs.size())[-3], list(relation_pairs.size())[-2],
                                         list(relation_pairs.size())[-1])
    # print(list(relation_pairs.size())) # [7600, 256, 5, 5]

    relations = relation_network(relation_pairs)
    #print(list(relations.size())) # [9000, 1]
    relations = relations.view(-1, n_way)
    #print(list(relations.size())) # [1000, 9]

    # 得到预测标签
    _, predict_label = torch.max(relations.data, 1)
    # print('predict_label', predict_label)

    return predict_label


def test(im_width, im_height, channels, jishujun):
    print()
    print()
    print()
    print("\033[0;33;40m{}\033[0m".format('*******************************************'))
    print("\033[0;33;40mtest! load {} support!\033[0m".format(jishujun))
    print("\033[0;33;40m{}\033[0m".format('*******************************************'))
    
    # 训练数据集 support
    load_train_dataset_name = \
    '../h5dataset_ica_bandselect_200/' + \
    h5datasetname + '_' +\
    str(im_width) + '_' + \
    str(im_height) + '_' + \
    str(depth) + '_support' + \
    str(n_examples) + '_' + \
    str(jishujun) + \
    '.h5'

    print("\033[0;33;40m{}\033[0m".format(str("check the file!")))
    print("\033[0;33;40m{}\033[0m".format(load_train_dataset_name))

    f = h5py.File(load_train_dataset_name, 'r')

    support_images = np.array(f['data_s'])  # (5, 8100)
    support_images = support_images.reshape(-1, im_width, im_height, channels).transpose((0, 3, 1, 2))[:, np.newaxis, :, :, :]
    print('support_images = ', support_images.shape)  # (9, 1, 100, 9, 9)
    f.close()



    # 测试数据集 test
    load_train_dataset_name = \
    '../h5dataset_ica_bandselect_200/' + \
    h5datasetname + '_' +\
    str(im_width) + '_' + \
    str(im_height) + '_' + \
    str(depth) + '_test' \
    + '.h5'

    print("\033[0;33;40m{}\033[0m".format(str("check the file!")))
    print("\033[0;33;40m{}\033[0m".format(load_train_dataset_name))

    f = h5py.File(load_train_dataset_name, 'r')

    test_images = np.array(f['data'])  # (42776, 8100)
    test_images = test_images.reshape(-1, im_width, im_height, channels).transpose((0, 3, 1, 2))[:, np.newaxis, :, :, :]
    print('test_images = ', test_images.shape)  # (42776, 1, 100, 9, 9)
    test_labels = f['label'][:]  # (42776, )
    f.close()

    predict_labels = []  # 记录预测标签
    # S1
    for i in range(0, int(n_samples/10)):#10988 42776 10249 54129
        # print('i=', i)
        # 54110 - 54120
        test_images_ = test_images[10 * i:10 * (i + 1), :, :, :, :]
        predict_label = rn_predict(support_images, test_images_, num = 10)
        predict_labels.extend(predict_label.cpu().numpy().tolist())

    # S2
    # 54120 - 54129
    rest_n = n_samples - (int(n_samples/10) * 10)
    test_images_ = test_images[-rest_n:, :, :, :, :]
    predict_label = rn_predict(support_images, test_images_, num = rest_n)
    predict_labels.extend(predict_label.cpu().numpy().tolist())

    # S3
    # print("test++=========")
    print(test_labels.shape,np.array(predict_labels).shape) # (42776,)
    # print(np.unique(predict_labels))
    rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(test_images.shape[0])]

    ##################### 混淆矩阵 #####################
    matrix = metrics.confusion_matrix(test_labels, predict_labels)
    # print(matrix)
    OA = np.sum(np.trace(matrix)) / 54129.0 * 100

    print("\033[0;33;40mOA = {}\033[0m".format(round(OA, 2)))

################################################################

    matrix = np.zeros((n_way, n_way), dtype=np.int)    # SA 16类标签 
    print("predict_labels LEN =", len(predict_labels))

    # 标签写入到CSV中
    nowtime = time.localtime()
    nowtime_style = time.strftime("%Y-%m-%d__%H_%M_%S", nowtime)
    path = './result/' + h5datasetname + '_label_test_predict_' + nowtime_style +'.csv'
    label_csv = open(path, 'w', encoding='utf-8', newline='')
    
    label_csv_writer = csv.writer(label_csv)
    label_csv_writer.writerow(["test label", "predict label"])
    for j in range(n_samples):
        matrix[test_labels[j], predict_labels[j]] += 1  # 构建混淆矩阵
        label_csv_writer.writerow([str(test_labels[j]), str(predict_labels[j])])
    label_csv.close()
    
    # print(matrix)
    # print(np.sum(np.trace(matrix)))  # np.trace 对角线元素之和

    print("\033[0;33;40mOA = {}\033[0m".format(np.sum(np.trace(matrix)) / float(n_samples) * 100))

    kappa_true = metrics.cohen_kappa_score(test_labels, predict_labels)

    kappa_temp, aa_temp, ca_temp = kappa(matrix, n_way) # 16代表 SA的标记样本类别数量

    print("\033[0;33;40mkappa = {}\033[0m".format(kappa_temp * 100))

    f = open('./result/' + h5datasetname + '_' + nowtime_style +'.txt', 'w')

    for index in range(len(ca_temp)):
        f.write(str(ca_temp[index]) + '\n')
    f.write(str(np.sum(np.trace(matrix)) / float(n_samples) * 100) + '\n')
    f.write(str(aa_temp) + '\n')
    f.write(str(kappa_true * 100) + '\n')
    
    result = \
    viz_tmp_name1 + '\n' + \
    'OA = ' + str(np.sum(np.trace(matrix)) / float(n_samples) * 100) + '\n' + \
    'AA = ' + str(aa_temp) + '\n' + \
    'KAPPA = ' + str(kappa_true * 100) + '\n'

    viz.text(result, win=viz_tmp_name1)

if __name__ == '__main__':
    if not os.path.exists('./model'):
        os.makedirs('./model')
    if not os.path.exists('./result'):
        os.makedirs('./result')
    if not os.path.exists('./log'):
        os.makedirs('log')
    make_print_to_file(path='./log', current_filename=sys.argv[0])

    test(im_width, im_height, depth, jishujun)
