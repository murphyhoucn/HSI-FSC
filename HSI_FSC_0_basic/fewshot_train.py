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

from utils import dataset_nway
from utils import make_print_to_file
from EM_RN_model import Embedding
from EM_RN_model import RelationNetwork
from EM_RN_model import weights_init

#torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-s","--n_shot",type = int, default = 1)       # support set per class
parser.add_argument("-b","--n_query",type = int, default = 4)       # query set per class

parser.add_argument("-me","--meta_episode",type = int, default= 10000)
parser.add_argument("-fe","--fewshot_episode",type = int, default= 1000)

parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)

parser.add_argument("-dn","--datasetname", type=str, default='SA')
parser.add_argument("-n","--count", type=int, default=0)
args = parser.parse_args()

# Hyper Parameters

n_shot = args.n_shot
n_query = args.n_query
META_EPISODE = args.meta_episode
FEW_SHOT_EPISODE = args.fewshot_episode
LEARNING_RATE = args.learning_rate
h5datasetname = args.datasetname
GPU = args.gpu
jishujun = args.count

n_way, n_samples = dataset_nway(h5datasetname)
if n_way == 0:
    print("\033[0;33;40m{}\033[0m".format(str("wrong!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")))

print("\033[0;33;40m{}\033[0m".format(str("check it")))
print("\033[0;33;40mdataset: {}\033[0m".format(h5datasetname))
print("\033[0;33;40mn_way: {}\033[0m".format(n_way))
print("\033[0;33;40mn_samples: {}\033[0m".format(n_samples))

n_examples = 5  # 训练数据集中每类5个样本
im_width, im_height, depth = 28, 28, 100 # 输入的cube为固定值

def fewshot_train(im_width, im_height, depth, jishujun):
    
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
    viz_tmp_name1 = 'fewshot_' + h5datasetname +'_loss'
    viz_tmp_name2 = 'fewshot_' + h5datasetname +'_acc'
    viz.line([[0.5]], [0], win= viz_tmp_name1, opts=dict(title=viz_tmp_name1, legend=[viz_tmp_name1]))
    viz.line([[0.]], [0], win=viz_tmp_name2, opts=dict(title=viz_tmp_name2, legend=[viz_tmp_name2]))

    feature_encoder = Embedding()
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



    # 载入元学习的模型
    model_name = './model'
    model_dir_file = os.listdir(model_name)
    model_dir_file_meta = [i for i in model_dir_file if i[0] == 'm']

    feature_encoder_model_name = model_name + '/' + model_dir_file_meta[0]
    relation_network_model_name = model_name + '/' + model_dir_file_meta[1]

    print("\033[0;33;40m{}\033[0m".format(str("check the file!")))
    print("\033[0;33;40m{}\033[0m".format(feature_encoder_model_name))
    print("\033[0;33;40m{}\033[0m".format(relation_network_model_name))

    feature_encoder.load_state_dict(torch.load(feature_encoder_model_name, map_location='cuda:0'))
    print("\033[0;33;40m{}\033[0m".format(str("load feature encoder success")))

    relation_network.load_state_dict(torch.load(relation_network_model_name, map_location='cuda:0'))
    print("\033[0;33;40m{}\033[0m".format(str("load relation network success")))

    # 进行 fewshot learning
    feature_encoder.train()
    relation_network.train()


    # 训练数据集 support
    load_train_dataset_name = \
    '../h5dataset_ica_bandselect_200/' + \
    h5datasetname + '_' + \
    str(im_width) + '_' + \
    str(im_height) + '_' + \
    str(depth) + '_support' + \
    str(n_examples) + '_' + str(jishujun) + '.h5'

    print("\033[0;33;40m{}\033[0m".format(str("check the file!")))
    print("\033[0;33;40m{}\033[0m".format(load_train_dataset_name))

    f = h5py.File(load_train_dataset_name, 'r')
    train_dataset = f['data_s'][:]
    f.close()
    train_dataset = train_dataset.reshape(-1, n_examples, im_width, im_height, depth) 
    train_dataset = train_dataset.transpose((0, 1, 4, 2, 3))[:, :, np.newaxis, :, :, :]
    # print(train_dataset.shape) # (16, 5, 1, 100, 28, 28)
    n_train_classes = train_dataset.shape[0]

    starttme = time.time()
    for episode in range(FEW_SHOT_EPISODE):

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

        ##### training #####
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
            viz.line([[disp_loss]], [episode+1], win=viz_tmp_name1, update='append')
            viz.line([[disp_acc]], [episode+1], win=viz_tmp_name2, update='append')

    print("few shot learning time : {}".format(time.time() - starttme))



    ##### Save model #####
    save_model_name1 = \
        './model/' +\
        h5datasetname + '_' +\
        'embdeeing_fewshot_' + str(FEW_SHOT_EPISODE) + 'fewshotepisode_' + str(jishujun) + '.pkl'
    
    print("\033[0;33;40m{}\033[0m".format(str("check the file!")))
    print("\033[0;33;40m{}\033[0m".format(save_model_name1))


    torch.save(
        feature_encoder.state_dict(),
        str(save_model_name1)
        )
    
    save_model_name2 = \
        './model/' +\
        h5datasetname + '_' +\
        'relation_fewshot_' + str(FEW_SHOT_EPISODE) + 'fewshotepisode_' + str(jishujun) + '.pkl'    
    torch.save(
        relation_network.state_dict(),
        str(save_model_name2)
        )
    
    print("\033[0;33;40m{}\033[0m".format(str("check the file!")))
    print("\033[0;33;40m{}\033[0m".format(save_model_name2))

    print("few shot model save success!")


if __name__ == '__main__':
    if not os.path.exists('./model'):
        os.makedirs('./model')
    if not os.path.exists('./result'):
        os.makedirs('./result')
    if not os.path.exists('./log'):
        os.makedirs('./log')
    make_print_to_file(path='./log', current_filename=sys.argv[0])

    fewshot_train(im_width, im_height, depth, jishujun)
