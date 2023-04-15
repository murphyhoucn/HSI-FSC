import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import argparse
import h5py
import time
import os
import visdom
import sys

from EM_RN_model import Embedding
from EM_RN_model import RelationNetwork
from EM_RN_model import weights_init
from utils import make_print_to_file

## cuDNN
#torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True

## visdom
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
viz.line([[0.5]], [0], win='meta_training_loss', opts=dict(title='meta_training_loss', legend=['meta_training_loss']))
viz.line([[0.]], [0], win='meta_training_acc', opts=dict(title='meta_training_acc', legend=['meta_training_acc']))

## input parameters/hyperparameters
parser = argparse.ArgumentParser(description="One Shot Visual Recognition")

parser.add_argument("-w","--n_way",type = int, default = 20)                      # way
parser.add_argument("-s","--n_shot",type = int, default = 1)       # support set per class
parser.add_argument("-b","--n_query",type = int, default = 19)       # query set per class

parser.add_argument("-me","--meta_episode",type = int, default= 10000)
parser.add_argument("-fe","--fewshot_episode",type = int, default= 1000)

parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
args = parser.parse_args()

n_way = args.n_way
n_shot = args.n_shot
n_query = args.n_query
META_EPISODE = args.meta_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu



n_examples = 200  # 训练数据集中每类200个样本
im_width, im_height, depth = 28, 28, 100 # 输入的cube为固定值

## train function
def meta_train(im_width, im_height, depth):
    print("n_way = {}, n_shot = {}, n_query = {}, META_EPISODE = {}, LEARNING_RATE = {}, GPU = {}".format(n_way, n_shot, n_query, META_EPISODE, LEARNING_RATE, GPU))
    print("n_examples = {}, im_width = {}, im_height = {}, depth = {}".format(n_examples, im_width, im_height, depth))
    ## neural network declaration
    feature_encoder = Embedding()
    relation_network = RelationNetwork()

    feature_encoder.apply(weights_init)
    relation_network.apply(weights_init)

    feature_encoder.cuda(GPU)
    relation_network.cuda(GPU)
    
    ## train model
    feature_encoder.train()
    relation_network.train()

    ## optimimizer
    ## Adam优化器，和lr更新策略
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(optimizer=feature_encoder_optim, step_size=5000, gamma=0.5)
    ## 每过step_size次,更新一次学习率;每经过100000次，学习率折半
    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim, step_size=5000, gamma=0.5)

    ## loat training dataset
    h5_path = '../h5dataset_ica_bandselect_200/'
    h5_path_dir_file = os.listdir(h5_path)
    meta_train_dataset_name = [i for i in h5_path_dir_file if i[0] == 'm']
    meta_train_dataset_name = h5_path + '/' + meta_train_dataset_name[0]

    print("\033[0;33;40m{}\033[0m".format(str("check the file!")))
    print("\033[0;33;40m{}\033[0m".format(meta_train_dataset_name))

    f = h5py.File(meta_train_dataset_name, 'r')
    train_dataset = f['data'][:]
    f.close()


    train_dataset = train_dataset.reshape(-1, n_examples, im_width, im_height, depth)
    # print(train_dataset.shape) # (55, 200, 28, 28, 100)
    train_dataset = train_dataset.transpose((0, 1, 4, 2, 3))[:, :, np.newaxis, :, :, :]
    # print(train_dataset.shape) # (55, 200, 1, 100, 28, 28)
    n_train_classes = train_dataset.shape[0]
    # print(n_train_classes) # 55

    ## training
    accuracy_ = []
    loss_ = []
    start_time = time.time()

    for episode in range(META_EPISODE):

        feature_encoder_scheduler.step(episode)
        relation_network_scheduler.step(episode)

        # start:每一个episode的采样过程#########
        epi_classes = np.random.permutation(n_train_classes)[:n_way]  # 生成[0，n_train_classes)的序列，然后随机打乱，取其中的前n_way个 
        # n_train_classes为类别数量 随机抽取n_way个类别
        # print(epi_classes) # [ 7 45 54 28  9 11 46 43 19 33 25 47 50 18  0 27  3 29  6 26]

        support = np.zeros([n_way, n_shot, 1, depth, im_height, im_width], dtype=np.float32)  # n_shot = 5
        query = np.zeros([n_way, n_query,  1, depth, im_height, im_width], dtype=np.float32)  # n_query= 15
        # print(support.shape, query.shape)
        # (20, 1, 1, 100, 28, 28) (20, 19, 1, 100, 28, 28)
        # (N,C_in,D_in,H_in,W_in)

        for i, epi_cls in enumerate(epi_classes):
            selected = np.random.permutation(n_examples)[:n_shot + n_query] # 支撑集合
            support[i] = train_dataset[epi_cls, selected[:n_shot]]
            query[i] = train_dataset[epi_cls, selected[n_shot:]]

        support = support.reshape(n_way * n_shot, 1, depth, im_height, im_width)
        query = query.reshape(n_way * n_query, 1, depth, im_height, im_width)
        labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query)).astype(np.uint8).reshape(-1)
        # print(labels.shape)# (380,)
        # print(labels) 

        support_tensor = torch.from_numpy(support)
        query_tensor = torch.from_numpy(query)
        label_tensor = torch.LongTensor(labels)
        # end:每一个episode的采样过程######

        # calculate features
        sample_features = feature_encoder(Variable(support_tensor).cuda(GPU))  # 数量*通道*高度*宽度
        # print("sample_features:", sample_features.shape) # sample_features: torch.Size([20, 64, 2, 5, 5])

        sample_features = sample_features.view(
            n_way, 
            n_shot, 
            list(sample_features.size())[-4], list(sample_features.size())[-3],
            list(sample_features.size())[-2], list(sample_features.size())[-1]
        )  # view函数改变shape


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
        #print(list(sample_features.size())) # [20, 128, 5, 5]
        #print(list(batch_features.size())) # [380, 128, 5, 5]
        ################################################################################################################

        # calculate relations
        # 支撑样本和查询样本进行连接
        #print('relation_pairs.size() = ',list(relation_pairs.size()))  # [6000, 384, 3, 3]


        sample_features_ext = sample_features.repeat(n_query * n_way, 1, 1, 1, 1)  # # repeat函数沿着指定的维度重复tensor
        #print(list(sample_features_ext.size())) # [380, 20, 128, 5, 5]
        batch_features_ext = batch_features.repeat(n_way, 1, 1, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
        #print(list(batch_features_ext.size())) # [380, 20, 128, 5, 5]

        relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2)
        #print(list(relation_pairs.size())) # [380, 20, 256, 5, 5]
        relation_pairs = relation_pairs.view(-1,  list(relation_pairs.size())[-3], list(relation_pairs.size())[-2], list(relation_pairs.size())[-1])
        #print(list(relation_pairs.size())) # [7600, 256, 5, 5]

        relations = relation_network(relation_pairs)
        # flops, params = profile(relation_network, inputs=(relation_pairs,))
        # print('relation_network flops', flops, 'params', params)
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

        if (episode + 1) % 20 == 0:
            print("episode:",episode + 1,"loss",loss)
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
            accuracy_.append(accuracy)
            loss_.append(loss.item())

            ############ visdom 显示 loss 和 acc #################
            disp_loss = loss.cpu()
            # print(type(disp_loss), disp_loss.device)
            disp_loss= disp_loss.detach().numpy()
            # print(type(disp_loss))
            disp_loss = disp_loss.astype(np.float64)
            # print(type(disp_loss))
            disp_acc = accuracy
            viz.line([[disp_loss]], [episode+1], win='meta_training_loss', update='append')
            viz.line([[disp_acc]], [episode+1], win='meta_training_acc', update='append')
            
            time.sleep(0.1)
            
    print('meta_train_time = ',time.time() - start_time)

    meta_training_embedding_name = str(
        './model/meta_training_embedding_' + \
        str(n_way) + 'way_' + \
        str(n_shot) + 'shot_' + \
        str(n_query) + 'querty_' + \
        str(LEARNING_RATE) + 'learning_' + \
        str(META_EPISODE) + 'metatrainepisode' + \
        '.pkl'
    )

    torch.save(feature_encoder.state_dict(), meta_training_embedding_name)
    
    meta_training_relation_name = str(
        './model/meta_training_relation_' + 
        str(n_way) + 'way_' + 
        str(n_shot) + 'shot_' + 
        str(n_query) + 'querty_' + 
        str(LEARNING_RATE) + 'learning_' + 
        str(META_EPISODE) + 'metatrainepisode' + 
        '.pkl'
    )
    torch.save(relation_network.state_dict(), meta_training_relation_name)

    meta_training_loss_name = \
    './result/meta_training_loss_' + \
        str(n_way) + 'way_' + \
        str(n_shot) + 'shot_' + \
        str(n_query) + 'querty_' + \
        str(LEARNING_RATE) + 'learning_' + \
        str(META_EPISODE) + 'metatrainepisode' + \
        '.txt'
    
    f = open(meta_training_loss_name, 'w')

    for i in range(np.array(loss_).shape[0]):
        f.write(str(loss_[i]) + '\n')
    
    meta_training_accuracy_name = \
        './result/meta_training_accuracy_' + \
        str(n_way) + 'way_' + \
        str(n_shot) + 'shot_' + \
        str(n_query) + 'querty_' + \
        str(LEARNING_RATE) + 'learning_' + \
        str(META_EPISODE) + 'metatrainepisode' + \
        '.txt'   
    
    f = open(meta_training_accuracy_name, 'w')

    for i in range(np.array(accuracy_).shape[0]):
        f.write(str(accuracy_[i]) + '\n')


# main funtion
if __name__ == '__main__':

    if not os.path.exists('./model'):
        os.makedirs('./model')
    if not os.path.exists('./result'):
        os.makedirs('./result')
    if not os.path.exists('./log'):
        os.makedirs('./log')
    make_print_to_file(path='./log', current_filename=sys.argv[0])

    meta_train(im_width, im_height, depth)
