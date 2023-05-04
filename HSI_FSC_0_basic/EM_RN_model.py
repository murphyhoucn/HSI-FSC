import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Embedding section
class Embedding(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(Embedding, self).__init__()

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
        
        # self.layer3_conv_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1))

        # self.layer3_conv_2_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1))
        # self.layer3_conv_2_2 = nn.Conv3d(16, 16, (3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1))

        # self.layer3_conv_3_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1))
        # self.layer3_conv_3_2 = nn.Conv3d(16, 16, (5, 5, 5), padding=(2, 2, 2), stride=(1, 1, 1))

        # self.layer3_conv_4_1 = nn.MaxPool3d(kernel_size=(3, 3, 3),padding=(1, 1, 1), stride=(1, 1, 1))
        # self.layer3_conv_4_2 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1))
        
        # self.layer4_conv_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1))

        # self.layer4_conv_2_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1))
        # self.layer4_conv_2_2 = nn.Conv3d(16, 16, (3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1))

        # self.layer4_conv_3_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1))
        # self.layer4_conv_3_2 = nn.Conv3d(16, 16, (5, 5, 5), padding=(2, 2, 2), stride=(1, 1, 1))

        # self.layer4_conv_4_1 = nn.MaxPool3d(kernel_size=(3, 3, 3),padding=(1, 1, 1), stride=(1, 1, 1))
        # self.layer4_conv_4_2 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1))

        self.layer5 = nn.Sequential(
                        nn.Conv3d(16,64,kernel_size=3,padding=1),
                        nn.BatchNorm3d(64),
                        nn.ReLU(),
                        nn.MaxPool3d(kernel_size=(4, 2, 2),padding=1))
        
        # self.layer6 = nn.Sequential(
        #                 nn.Conv3d(32,64,kernel_size=3,padding=1),
        #                 nn.BatchNorm3d(64),
        #                 nn.ReLU())

    def forward(self,x):
        # print("embedding network section:")
        # print('size x:', list(x.size()))

        out = self.layer1(x)
        # print('out = layer1(x)\nsize out:', list(out.size())) # size out: [20, 8, 25, 15, 15]
          
        out = self.layer2(out)
        # print('out = layer2(x)\nsize out:', list(out.size()))  # size out: [20, 16, 6, 8, 8]

        # out1_layer3 = self.layer3_conv_1(out)
        # # print(out1_layer3.size())
        # out2_layer3 = self.layer3_conv_2_2(self.layer3_conv_2_1(out))
        # # print(out2_layer3.size())
        # out3_layer3 = self.layer3_conv_3_2(self.layer3_conv_3_1(out))
        # # print(out3_layer3.size())
        # out4_layer3 = self.layer3_conv_4_2(self.layer3_conv_4_1(out))
        # # print(out4_layer3.size())

        # out = F.relu(out1_layer3 + out2_layer3 + out3_layer3 + out4_layer3)
        # print('layer3\nsize out:', list(out.size()))  # size out: [20, 16, 6, 8, 8]

        # out1_layer4 = self.layer4_conv_1(out)
        # # print(out1_layer4.size())
        # out2_layer4 = self.layer3_conv_2_2(self.layer4_conv_2_1(out))
        # # print(out2_layer4.size())
        # out3_layer4 = self.layer3_conv_3_2(self.layer4_conv_3_1(out))
        # # print(out3_layer4.size())
        # out4_layer4 = self.layer3_conv_4_2(self.layer4_conv_4_1(out))
        # # print(out4_layer4.size())

        # out = F.relu(out1_layer4 + out2_layer4 + out3_layer4 + out4_layer4)
        # print('layer4\nsize out:', list(out.size()))  # size out: [20, 16, 6, 8, 8]

        out = self.layer5(out)
        # print('out = layer5(x)\nsize out:', list(out.size()))  # size out: [20, 32, 2, 5, 5]
        
        # out = self.layer6(out)
        # print('out = layer6(x)\nsize out:', list(out.size()))  # size out: [20, 64, 2, 5, 5]

        return out # size out: [20, 64, 2, 5, 5]

# Relation section
class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self):
        super(RelationNetwork, self).__init__()

        self.layer1 = nn.Sequential(
                        nn.Conv2d(256, 128,kernel_size=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU())

        self.layer2 = nn.Sequential(
                        nn.Conv2d(128, 64, kernel_size=3, padding=0),
                        nn.BatchNorm2d(64),
                        nn.ReLU())

        self.mp = nn.MaxPool2d(kernel_size=(3, 3),padding=0)

        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(p = 0.5) 

    def forward(self,x): 
        # print("relation network section:")
        # print('size x:', list(x.size()))

        out = self.layer1(x)
        # print('out = layer1(x)\nsize out:', list(out.size()))

        out = self.layer2(out)
        # print('out = layer2(x)\nsize out:', list(out.size()))

        out = self.mp(out)
        # print('out -> mp\nsize out:', list(out.size()))

        out = out.view(out.size(0),-1) # flatten
        # print('out -> flatten\nsize out:', list(out.size()))

        out = F.relu(self.fc1(out))
        # print('out -> fc1\nsize out:', list(out.size()))

        out = self.dropout(out)
        out = torch.sigmoid(self.fc2(out))
        # print('out -> fc2\nsize out:', list(out.size()))

        return out

# net init
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
