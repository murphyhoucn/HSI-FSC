import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import math
import argparse
import h5py
import time
from sklearn import metrics
import csv

#torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True

def kappa(testData, k): #testData表示要计算的数据，k表示数据矩阵的是k*k的
    dataMat = np.mat(testData)
    s = dataMat.sum()
    #print(dataMat.shape)
    print(dataMat)
    P0 = 0.0
    for i in range(k):
        P0 += dataMat[i, i]*1.0
    xsum = np.sum(dataMat, axis=1)
    ysum = np.sum(dataMat, axis=0)
    #xsum是个k行1列的向量，ysum是个1行k列的向量
    #Pe = float(ysum * xsum) / float(s * 1.0) / float(s * 1.0)
    Pe = float(ysum * xsum) / float(s ** 2)
    print("Pe = ", Pe)
    P0 = float(P0/float(s*1.0))
    #print("P0 = ", P0)
    cohens_coefficient = float((P0-Pe)/(1-Pe))

    a = []
    a = dataMat.sum(axis=0)
    a = np.float32(a)
    a = np.array(a)
    a = np.squeeze(a)

    print(a)

    for i in range(k):
        #print(dataMat[i, i])
        a[i] = float(dataMat[i, i]*1.0)/float(a[i]*1.0)
    print(a*100)
    #print(a.shape)
    print("AA: ", a.mean()*100)
    return cohens_coefficient, a.mean()*100, a*100

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
#parser.add_argument("-f","--feature_dim",type = int, default = 256)              # 最后一个池化层输出的维度
#parser.add_argument("-r","--relation_dim",type = int, default = 8)               # 第一个全连接层维度
parser.add_argument("-w","--n_way",type = int, default = 9)                      # way  XZ_nway
parser.add_argument("-s","--n_shot",type = int, default = 5)       # support set per class
#parser.add_argument("-b","--n_query",type = int, default = 19)       # query set per class
#parser.add_argument("-e","--episode",type = int, default= 1)
#-----------------------------------------------------------------------------------#
#parser.add_argument("-t","--test_episode", type = int, default = 600)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
args = parser.parse_args()


# Hyper Parameters
#FEATURE_DIM = args.feature_dim
#RELATION_DIM = args.relation_dim
n_way = args.n_way
n_shot = args.n_shot
#n_query = args.n_query
#EPISODE = args.episode
#-----------------------------------------------------------------------------------#
#TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu

#n_examples = 200  # 训练数据集中每类200个样本
im_width, im_height, channels = 28, 28, 100 # 输入的cube为固定值


class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()

        self.layer1 = nn.Sequential(
                        nn.Conv3d(1,8,kernel_size=3,padding=1),
                        # Conv3d(in_depth, out_depth, kernel_size, stride=1, padding=0)
                        nn.BatchNorm3d(8),
                        nn.ReLU(),
                        nn.MaxPool3d(kernel_size=(4, 2, 2),padding=1))

        self.layer2 = nn.Sequential(
                        nn.Conv3d(8,16,kernel_size=3,padding=1),
                        nn.BatchNorm3d(16),
                        nn.ReLU(),
                        nn.MaxPool3d(kernel_size=(4, 2, 2),padding=1))
        
        self.layer3_conv_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1))

        self.layer3_conv_2_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1))
        self.layer3_conv_2_2 = nn.Conv3d(16, 16, (3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1))

        self.layer3_conv_3_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1))
        self.layer3_conv_3_2 = nn.Conv3d(16, 16, (5, 5, 5), padding=(2, 2, 2), stride=(1, 1, 1))

        self.layer3_conv_4_1 = nn.MaxPool3d(kernel_size=(3, 3, 3),padding=(1, 1, 1), stride=(1, 1, 1))
        self.layer3_conv_4_2 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1))
        
        self.layer4_conv_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1))

        self.layer4_conv_2_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1))
        self.layer4_conv_2_2 = nn.Conv3d(16, 16, (3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1))

        self.layer4_conv_3_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1))
        self.layer4_conv_3_2 = nn.Conv3d(16, 16, (5, 5, 5), padding=(2, 2, 2), stride=(1, 1, 1))

        self.layer4_conv_4_1 = nn.MaxPool3d(kernel_size=(3, 3, 3),padding=(1, 1, 1), stride=(1, 1, 1))
        self.layer4_conv_4_2 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1))

        self.layer5 = nn.Sequential(
                        nn.Conv3d(16,32,kernel_size=3,padding=1),
                        nn.BatchNorm3d(32),
                        nn.ReLU(),
                        nn.MaxPool3d(kernel_size=(4, 2, 2),padding=1))
        
        self.layer6 = nn.Sequential(
                        nn.Conv3d(32,64,kernel_size=3,padding=1),
                        nn.BatchNorm3d(64),
                        nn.ReLU())

    def forward(self,x):
        # print("embedding network section:")
        # print('size x:', list(x.size()))

        out = self.layer1(x)
        # print('out = layer1(x)\nsize out:', list(out.size())) # size out: [20, 8, 25, 15, 15]
          
        out = self.layer2(out)
        # print('out = layer2(x)\nsize out:', list(out.size()))  # size out: [20, 16, 6, 8, 8]

        out1_layer3 = self.layer3_conv_1(out)
        # print(out1_layer3.size())
        out2_layer3 = self.layer3_conv_2_2(self.layer3_conv_2_1(out))
        # print(out2_layer3.size())
        out3_layer3 = self.layer3_conv_3_2(self.layer3_conv_3_1(out))
        # print(out3_layer3.size())
        out4_layer3 = self.layer3_conv_4_2(self.layer3_conv_4_1(out))
        # print(out4_layer3.size())

        out = F.relu(out1_layer3 + out2_layer3 + out3_layer3 + out4_layer3)
        # print('layer3\nsize out:', list(out.size()))  # size out: [20, 16, 6, 8, 8]

        out1_layer4 = self.layer4_conv_1(out)
        # print(out1_layer4.size())
        out2_layer4 = self.layer3_conv_2_2(self.layer4_conv_2_1(out))
        # print(out2_layer4.size())
        out3_layer4 = self.layer3_conv_3_2(self.layer4_conv_3_1(out))
        # print(out3_layer4.size())
        out4_layer4 = self.layer3_conv_4_2(self.layer4_conv_4_1(out))
        # print(out4_layer4.size())

        out = F.relu(out1_layer4 + out2_layer4 + out3_layer4 + out4_layer4)
        # print('layer3\nsize out:', list(out.size()))  # size out: [20, 16, 6, 8, 8]

        out = self.layer5(out)
        # print('out = layer3(x)\nsize out:', list(out.size()))  # size out: [20, 32, 2, 5, 5]
        
        out = self.layer6(out)
        # print('out = layer4(x)\nsize out:', list(out.size()))  # size out: [20, 64, 2, 5, 5]

        return out # size out: [20, 64, 2, 5, 5]

# Relation section
class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self):
        super(RelationNetwork, self).__init__()

        self.layer1 = nn.Sequential(
                        nn.Conv2d(256, 128,kernel_size=3),
                        nn.BatchNorm2d(128),
                        nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(p = 0.5)    # 测试的时候需要修改....？？？

    def forward(self,x): 
        # print("relation network section:")
        # print('size x:', list(x.size()))

        out = self.layer1(x)
        # print('out = layer1(x)\nsize out:', list(out.size()))

        out = self.layer2(out)
        # print('out = layer2(x)\nsize out:', list(out.size()))

        out = out.view(out.size(0),-1) # flatten
        # print('out -> flatten\nsize out:', list(out.size()))

        out = F.relu(self.fc1(out))
        # print('out -> fc1\nsize out:', list(out.size()))

        out = self.dropout(out)
        out = torch.sigmoid(self.fc2(out))
        # print('out -> fc2\nsize out:', list(out.size()))

        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


feature_encoder = CNNEncoder()
relation_network = RelationNetwork()

feature_encoder.cuda(GPU)
relation_network.cuda(GPU)

feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=100000, gamma=0.5)
relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
relation_network_scheduler = StepLR(relation_network_optim, step_size=100000, gamma=0.5)


feature_encoder.load_state_dict(torch.load(str("../model/XZ_feature_encoder_20way_1shot_newmodel_5FT_1000.pkl")))
print("load feature encoder success")

relation_network.load_state_dict(torch.load(str("../model/XZ_relation_network_20way_1shot_newmodel_5FT_1000.pkl")))
print("load relation network success")

feature_encoder.eval()
relation_network.eval()

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


def test(im_width, im_height, channels):

    #A = time.time()
    # 加载支撑数据
    f = h5py.File('../h5dataset_ica_bandselect_200/XZ_' + str(im_width) + '_' + str(im_height) + '_' + str(channels) + '_support' + str(args.n_shot) + '.h5', 'r')
    support_images = np.array(f['data_s'])  # (5, 8100)
    support_images = support_images.reshape(-1, im_width, im_height, channels).transpose((0, 3, 1, 2))[:, np.newaxis, :, :, :]
    print('support_images = ', support_images.shape)  # (9, 1, 100, 9, 9)
    f.close()

    # 加载测试
    f = h5py.File(r'../h5dataset_ica_bandselect_200/XZ_28_28_100_test.h5', 'r')  # 路径
    test_images = np.array(f['data'])  # (42776, 8100)
    test_images = test_images.reshape(-1, im_width, im_height, channels).transpose((0, 3, 1, 2))[:, np.newaxis, :, :, :]
    print('test_images = ', test_images.shape)  # (42776, 1, 100, 9, 9)
    test_labels = f['label'][:]  # (42776, )
    f.close()

    #epi_classes = np.random.permutation(test_images.shape[0])
    #test_images = test_images[epi_classes, :, :, :, :]
    #test_labels = test_labels[epi_classes]

    predict_labels = []  # 记录预测标签
    # S1
    for i in range(0, 6887):#10988 42776 10249 54129
        # 68860 - 68870
        # print('i=', i)
        test_images_ = test_images[10 * i:10 * (i + 1), :, :, :, :]
        predict_label = rn_predict(support_images, test_images_, num = 10)
        predict_labels.extend(predict_label.cpu().numpy().tolist())

    # S2
    # 68870 -68877
    test_images_ = test_images[-7:, :, :, :, :]
    predict_label = rn_predict(support_images, test_images_, num = 7)
    predict_labels.extend(predict_label.cpu().numpy().tolist())

    # S3
    #print(test_labels.shape) # (42776,)
    print(np.unique(predict_labels))
    #print(np.array(predict_labels).shape) # (42776,)
    rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(test_images.shape[0])]

    ##################### 混淆矩阵 #####################
    from sklearn import metrics
    matrix = metrics.confusion_matrix(test_labels, predict_labels)
    print(matrix)
    OA = np.sum(np.trace(matrix)) / 68877.0 * 100
    print('OA = ', round(OA, 2))

    #print("kappa = ", round(kappa(matrix, 9) * 100, 2))

    # print(rewards)
    #total_rewards = np.sum(rewards)
    # print(total_rewards)

    #print(time.time()-A)

    #accuracy = total_rewards / test_images.shape[0]
    #print("accuracy:", accuracy)

    # f = open('./prediction_' + str(round(OA, 2)) + '.txt', 'w')
    # for i in range(test_images.shape[0]):
    #     f.write(str(predict_labels[i]) + '\n')



################################################################
    # n = 54129  # SA的测试标签数量 16 类
    # n = 148152 # PC 9 类
    # n = 42776 # UP  9类
    n = 68877 # XZ  9类
    matrix = np.zeros((9, 9), dtype=np.int)  # 9类
    print("predict_labels LEN =", len(predict_labels))


    # 标签写入到CSV中
    nowtime = time.localtime()
    nowtime_style = time.strftime("%Y-%m-%d %H:%M:%S", nowtime)
    path = '../result/' + 'XZ_label_test_predict' + nowtime_style +'.csv'
    label_csv = open(path, 'w', encoding='utf-8', newline='')

    label_csv_writer = csv.writer(label_csv)
    label_csv_writer.writerow(["test label", "predict label"])
    for j in range(n):
        matrix[test_labels[j], predict_labels[j]] += 1  # 构建混淆矩阵
        label_csv_writer.writerow([str(test_labels[j]), str(predict_labels[j])])
    label_csv.close()
    
    # print(matrix)
    # print(np.sum(np.trace(matrix)))  # np.trace 对角线元素之和
    print("OA: ", np.sum(np.trace(matrix)) / float(n) * 100)

    from sklearn import metrics
    kappa_true = metrics.cohen_kappa_score(test_labels, predict_labels)

    kappa_temp, aa_temp, ca_temp = kappa(matrix, 9) # 代表 XZ的标记样本类别数量
    print(kappa_temp * 100)
    f = open('../result/XZ_' + nowtime_style + '.txt', 'w')
    for index in range(len(ca_temp)):
        f.write(str(ca_temp[index]) + '\n')
    f.write(str(np.sum(np.trace(matrix)) / float(n) * 100) + '\n')
    f.write(str(aa_temp) + '\n')
    f.write(str(kappa_true * 100) + '\n')




    # from scipy.io import loadmat
    # gt = loadmat('../dataset/13_Xuzhou/xuzhou_gt.mat')['xuzhou_gt']

    # # # 将预测的结果匹配到图像中
    # new_show = np.zeros((gt.shape[0], gt.shape[1]))
    # k = 0
    # for i in range(gt.shape[0]):
    #     for j in range(gt.shape[1]):
    #         if gt[i][j] != 0:
    #             new_show[i][j] = predict_labels[k]
    #             new_show[i][j] += 1
    #             k += 1

    # # print new_show.shape

    # # 展示地物
    # import matplotlib as mpl
    # import matplotlib.pyplot as  plt

    # XZ_colors = ['black', 'gray', 'lime', 'cyan', 'forestgreen', 'hotpink', 'saddlebrown',
    #           'purple', 'red', 'yellow', 'blue', 'steelblue', 'olive', 'sandybrown', 'mediumaquamarine', 'darkorange',
    #           'whitesmoke']

    # # colors = ['gray', 'lime', 'cyan', 'forestgreen', 'hotpink', 'saddlebrown', 'purple', 'red', 'yellow']
    # cmap = mpl.colors.ListedColormap(XZ_colors)
    # plt.xticks([])
    # plt.yticks([])
    # plt.imshow(new_show, cmap=cmap)
    # plt.savefig("../result/XZ_" + str(str(np.sum(np.trace(matrix)) / float(n) * 100)) + ".png", dpi=1000)  # 保存图像
    # # plt.savefig("predict_all.png")#保存图像


if __name__ == '__main__':
    test(im_width, im_height, channels)
