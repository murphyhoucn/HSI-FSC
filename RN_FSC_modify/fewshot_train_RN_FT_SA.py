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

#torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True

import visdom
viz = visdom.Visdom()
viz.line([[0.5]], [0], win='fewshot_SA_loss', opts=dict(title='fewshot_SA_loss', legend=['fewshot_SA_loss']))
viz.line([[0.]], [0], win='fewshot_SA_acc', opts=dict(title='fewshot_SA_acc', legend=['fewshot_SA_acc']))



parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
#parser.add_argument("-f","--feature_dim",type = int, default = 512)              # 最后一个池化层输出的维度
#parser.add_argument("-r","--relation_dim",type = int, default = 128)               # 第一个全连接层维度
parser.add_argument("-w","--n_way",type = int, default = 16)                      # way
parser.add_argument("-s","--n_shot",type = int, default = 1)       # support set per class
parser.add_argument("-b","--n_query",type = int, default = 4)       # query set per class

parser.add_argument("-e","--episode",type = int, default= 1000)   # episode 原来是1000，为了看过程，这里改成1
# parser.add_argument("-e","--episode",type = int, default= 1)   # episode 原来是1000，为了看过程，这里改成1

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
n_query = args.n_query
EPISODE = args.episode
#-----------------------------------------------------------------------------------#
#TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu

n_examples = 5  # 训练数据集中每类5个样本
im_width, im_height, depth = 28, 28, 100 # 输入的cube为固定值


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

def train(im_width, im_height, depth):

    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork()

    feature_encoder.apply(weights_init)
    relation_network.apply(weights_init)

    feature_encoder.cuda(GPU)
    relation_network.cuda(GPU)

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(optimizer=feature_encoder_optim, step_size=700, gamma=0.5)
    # 每过step_size次,更新一次学习率;每经过100000次，学习率折半
    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim, step_size=700, gamma=0.5)

    feature_encoder.load_state_dict(torch.load(str("../model/meta_training_feature_encoder_20way_1shot_newmodel.pkl"), map_location='cuda:0'))
    print("load feature encoder success")

    relation_network.load_state_dict(torch.load(str("../model/meta_training_relation_network_20way_1shot_newmodel.pkl"), map_location='cuda:0'))
    print("load relation network success")

    feature_encoder.train()
    relation_network.train()


    # 训练数据集
    f = h5py.File('../h5dataset_ica_bandselect_200/SA_' + str(im_width) + '_' + str(im_height) + '_' + str(depth) + '_support' + str(n_examples) + '.h5', 'r')
    train_dataset = f['data_s'][:]
    f.close()
    train_dataset = train_dataset.reshape(-1, n_examples, im_width, im_height, depth) 
    train_dataset = train_dataset.transpose((0, 1, 4, 2, 3))[:, :, np.newaxis, :, :, :]
    print(train_dataset.shape) # (16, 5, 1, 100, 28, 28)
    n_train_classes = train_dataset.shape[0]

    A = time.time()
    for episode in range(EPISODE):

        feature_encoder_scheduler.step(episode)
        relation_network_scheduler.step(episode)

        # start:每一个episode的采样过程##########################################################################################
        epi_classes = np.random.permutation(n_train_classes)[:n_way]  # 在78个数里面随机抽取前20个 78为类别数量 随机抽取20个类别，例如15 69 23 ....
        support = np.zeros([n_way, n_shot, 1, depth, im_height, im_width], dtype=np.float32)  # n_shot = 5
        query = np.zeros([n_way, n_query,  1, depth, im_height, im_width], dtype=np.float32)  # n_query= 15
        # (N,C_in,D_in,H_in,W_in)

        for i, epi_cls in enumerate(epi_classes):
            selected = np.random.permutation(n_examples)[:n_shot + n_query] # 支撑集合
            support[i] = train_dataset[epi_cls, selected[:n_shot]]
            query[i] = train_dataset[epi_cls, selected[n_shot:]]

        support = support.reshape(n_way * n_shot, 1, depth, im_height, im_width)
        query = query.reshape(n_way * n_query, 1, depth, im_height, im_width)
        labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query)).astype(np.uint8).reshape(-1)
        #print(labels)
        support_tensor = torch.from_numpy(support)
        query_tensor = torch.from_numpy(query)
        label_tensor = torch.LongTensor(labels)
        # end:每一个episode的采样过程##########################################################################################

        # calculate features
        sample_features = feature_encoder(Variable(support_tensor).cuda(GPU))  # 数量*通道*高度*宽度
        #print( list(sample_features.size()) ) # [100, 32, 6, 3, 3]
        sample_features = sample_features.view(n_way, n_shot, list(sample_features.size())[-4], list(sample_features.size())[-3],
                                               list(sample_features.size())[-2], list(sample_features.size())[
                                                   -1])  # view函数改变shape: 5way, 5shot, 64, 19, 19
        #sample_features = torch.sum(sample_features, 1).squeeze(1)  # 同类样本作和
        sample_features = torch.mean(sample_features, 1).squeeze(1)  # 同类样本取平均
        #print( list(sample_features.size()) ) # [20, 32, 6, 3, 3]
        batch_features = feature_encoder(Variable(query_tensor).cuda(GPU))  # 20x64*5*5
        #print(list(batch_features.size())) # [300, 32, 6, 3, 3]

        ################################################################################################################
        sample_features = sample_features.view(n_way, list(sample_features.size())[1]*list(sample_features.size())[2],
                                               list(sample_features.size())[-2], list(sample_features.size())[-1])
        batch_features = batch_features.view(n_way*n_query, list(batch_features.size())[1] * list(batch_features.size())[2],
                                               list(batch_features.size())[-2], list(batch_features.size())[-1])
        #print(list(sample_features.size())) # [20, 192, 3, 3]
        #print(list(batch_features.size())) # [300, 192, 3, 3]
        ################################################################################################################

        # calculate relations
        # 支撑样本和查询样本进行连接
        sample_features_ext = sample_features.repeat(n_query * n_way, 1, 1, 1, 1)  # # repeat函数沿着指定的维度重复tensor
        # print(list(sample_features_ext.size())) # [380, 20, 128, 5, 5]
        batch_features_ext = batch_features.repeat(n_way, 1, 1, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
        # print(list(batch_features_ext.size())) # [380, 20, 128, 5, 5]

        relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2)
        # print(list(relation_pairs.size())) # [380, 20, 256, 5, 5]
        relation_pairs = relation_pairs.view(-1, list(relation_pairs.size())[-3], list(relation_pairs.size())[-2],
                                             list(relation_pairs.size())[-1])
        # print(list(relation_pairs.size())) # [7600, 256, 5, 5]

        relations = relation_network(relation_pairs)
        #print(list(relations.size())) # [6000, 1]
        relations = relations.view(-1, n_way)
        #print(list(relations.size())) # [300, 20]

        mse = nn.MSELoss().cuda(GPU)
        one_hot_labels = Variable(
            torch.zeros(n_query * n_way, n_way).scatter_(dim=1, index=label_tensor.view(-1, 1), value=1).cuda(GPU))
        # scatter中1表示按照行顺序进行填充，labels_tensor.view(-1,1)为索引，1为填充数字
        loss = mse(relations, one_hot_labels)

        # training
        # 把模型中参数的梯度设为0
        feature_encoder.zero_grad()
        relation_network.zero_grad()

        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm(feature_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm(relation_network.parameters(), 0.5)

        # 进行单次优化，参数更新
        feature_encoder_optim.step()
        relation_network_optim.step()


        if (episode+1) % 10 == 0:
            print("episode:",episode+1,"loss",loss)
            #################调试#################
            _, predict_label = torch.max(relations.data, 1)
            predict_label = predict_label.cpu().numpy().tolist()
            #print(predict_label)
            #print(labels)
            rewards = [1 if predict_label[j] == labels[j] else 0 for j in range(labels.shape[0])]
            # print(rewards)
            total_rewards = np.sum(rewards)
            # print(total_rewards)

            accuracy = total_rewards*100.0 / labels.shape[0]
            print("accuracy:", accuracy)

            
            ############ visdom 显示 loss 和 acc #################
            disp_loss = loss.cpu()
            # print(type(disp_loss), disp_loss.device)
            disp_loss= disp_loss.detach().numpy()
            # print(type(disp_loss))
            disp_loss = disp_loss.astype(np.float64)
            # print(type(disp_loss))
            disp_acc = accuracy
            viz.line([[disp_loss]], [episode+1], win='fewshot_SA_loss', update='append')
            viz.line([[disp_acc]], [episode+1], win='fewshot_SA_acc', update='append')

    print(time.time()-A)

    torch.save(feature_encoder.state_dict(),str('../model/SA_feature_encoder_20way_1shot_newmodel_' + str(n_examples) + 'FT_' + str(EPISODE) + '.pkl'))
    torch.save(relation_network.state_dict(),str('../model/SA_relation_network_20way_1shot_newmodel_' + str(n_examples) + 'FT_' + str(EPISODE) + '.pkl'))

    print("model save success!")


if __name__ == '__main__':
    train(im_width, im_height, depth)
