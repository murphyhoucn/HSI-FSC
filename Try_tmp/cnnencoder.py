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
        print("embedding network section:")
        print('size x:', list(x.size()))

        out = self.layer1(x)
        print('out = layer1(x)\nsize out:', list(out.size())) # size out: [20, 8, 25, 15, 15]
          
        out = self.layer2(out)
        print('out = layer2(x)\nsize out:', list(out.size()))  # size out: [20, 16, 6, 8, 8]

        out1_layer3 = self.layer3_conv_1(out)
        print(out1_layer3.size())
        out2_layer3 = self.layer3_conv_2_2(self.layer3_conv_2_1(out))
        print(out2_layer3.size())
        out3_layer3 = self.layer3_conv_3_2(self.layer3_conv_3_1(out))
        print(out3_layer3.size())
        out4_layer3 = self.layer3_conv_4_2(self.layer3_conv_4_1(out))
        print(out4_layer3.size())

        out = F.relu(out1_layer3 + out2_layer3 + out3_layer3 + out4_layer3)
        print('layer3\nsize out:', list(out.size()))  # size out: [20, 16, 6, 8, 8]

        out1_layer4 = self.layer4_conv_1(out)
        print(out1_layer4.size())
        out2_layer4 = self.layer3_conv_2_2(self.layer4_conv_2_1(out))
        print(out2_layer4.size())
        out3_layer4 = self.layer3_conv_3_2(self.layer4_conv_3_1(out))
        print(out3_layer4.size())
        out4_layer4 = self.layer3_conv_4_2(self.layer4_conv_4_1(out))
        print(out4_layer4.size())

        out = F.relu(out1_layer3 + out2_layer3 + out3_layer3 + out4_layer3)
        print('layer3\nsize out:', list(out.size()))  # size out: [20, 16, 6, 8, 8]

        out = self.layer5(out)
        print('out = layer3(x)\nsize out:', list(out.size()))  # size out: [20, 32, 2, 5, 5]
        
        out = self.layer6(out)
        print('out = layer4(x)\nsize out:', list(out.size()))  # size out: [20, 64, 2, 5, 5]

        return out # size out: [20, 64, 2, 5, 5]
        
if __name__ == '__main__':

    net = nn.Sequential(
        CNNEncoder()
        )
    
    X = torch.rand(size=(20, 1, 100, 28, 28))

    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__,'output shape:\t', X.shape)