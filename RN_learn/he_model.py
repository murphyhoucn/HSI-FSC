import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.nn import init

# utils
import math
import os
import datetime
import numpy as np
import joblib



class HeEtAl(nn.Module):
    """
    MULTI-SCALE 3D DEEP CONVOLUTIONAL NEURAL NETWORK FOR HYPERSPECTRAL
    IMAGE CLASSIFICATION
    Mingyi He, Bo Li, Huahui Chen
    IEEE International Conference on Image Processing (ICIP) 2017
    https://ieeexplore.ieee.org/document/8297014/
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=7):
        super(HeEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(1, 16, (11, 3, 3), stride=(3, 1, 1))
        
        self.conv2_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0))
        self.conv2_2 = nn.Conv3d(16, 16, (3, 1, 1), padding=(1, 0, 0))
        self.conv2_3 = nn.Conv3d(16, 16, (5, 1, 1), padding=(2, 0, 0))
        self.conv2_4 = nn.Conv3d(16, 16, (11, 1, 1), padding=(5, 0, 0))
        
        self.conv3_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0))
        self.conv3_2 = nn.Conv3d(16, 16, (3, 1, 1), padding=(1, 0, 0))
        self.conv3_3 = nn.Conv3d(16, 16, (5, 1, 1), padding=(2, 0, 0))
        self.conv3_4 = nn.Conv3d(16, 16, (11, 1, 1), padding=(5, 0, 0))

        self.conv4 = nn.Conv3d(16, 16, (3, 2, 2))

        self.pooling = nn.MaxPool2d((3, 2, 2), stride=(3, 2, 2))

        # the ratio of dropout is 0.6 in our experiments
        self.dropout = nn.Dropout(p=0.6)

        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            # print(x.shape)
            x = self.conv1(x)
            # print(x.shape)

            x2_1 = self.conv2_1(x)
            # print("x2_1", x2_1.shape)
            x2_2 = self.conv2_2(x)
            # print("x2_2", x2_2.shape)
            x2_3 = self.conv2_3(x)
            # print("x2_3", x2_3.shape)
            x2_4 = self.conv2_4(x)
            # print("x2_4", x2_4.shape)

            x = x2_1 + x2_2 + x2_3 + x2_4
            # print(x.shape)

            x3_1 = self.conv3_1(x)
            x3_2 = self.conv3_2(x)
            x3_3 = self.conv3_3(x)
            x3_4 = self.conv3_4(x)

            x = x3_1 + x3_2 + x3_3 + x3_4
            # print(x.shape)

            x = self.conv4(x)
            # print(x.shape)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        print("x", x.shape)

        x = F.relu(self.conv1(x))
        print("F.relu(self.conv1(x)):", x.shape)

        x2_1 = self.conv2_1(x)
        print("x2_1 = self.conv2_1(x):", x2_1.shape)

        x2_2 = self.conv2_2(x)
        print("xxxxxxxxxxxxxxxxx2_2 = self.conv2_2(x):", x2_2.shape)

        x2_3 = self.conv2_3(x)
        print("x2_3 = self.conv2_3(x):", x2_3.shape)

        x2_4 = self.conv2_4(x)
        print("x2_4 = self.conv2_4(x):", x2_4.shape)

        x = x2_1 + x2_2 + x2_3 + x2_4
        print("x = x2_1 + x2_2 + x2_3 + x2_4:", x.shape)




        x = F.relu(x)
        print("x = F.relu(x):", x.shape)

        x3_1 = self.conv3_1(x)
        print("x3_1 = self.conv3_1(x)", x3_1.shape)
        x3_2 = self.conv3_2(x)
        print("x3_2 = self.conv3_1(x)", x3_2.shape)
        x3_3 = self.conv3_3(x)
        print("x3_3 = self.conv3_1(x)", x3_3.shape)
        x3_4 = self.conv3_4(x)
        print("x3_4 = self.conv3_1(x)", x3_4.shape)

        x = x3_1 + x3_2 + x3_3 + x3_4
        print("x3_1 = self.conv3_1(x)", x.shape)

        x = F.relu(x)
        print("x = F.relu(x)", x.shape)

        x = F.relu(self.conv4(x))
        print("x = F.relu(self.conv4(x))", x.shape)


        x = x.view(-1, self.features_size)
        print("x = x.view(-1, self.features_size)", x.shape)

        x = self.dropout(x)
        print("x = self.dropout(x)", x.shape)

        x = self.fc(x)
        print("x = self.fc(x)", x.shape)

        return x
    


if __name__ == '__main__':
    # net = HeEtAl(100, 10)
    # print(net)

    net = nn.Sequential(
        HeEtAl(100, 10)
    )

    X = torch.rand(size=(1, 1, 100, 7, 7))

    for layer in net:
        X = layer(X)
        # print(layer.__class__.__name__,'output shape:\t', X.shape)