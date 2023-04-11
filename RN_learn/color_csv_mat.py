import pandas as pd
import numpy as np
from scipy.io import loadmat
import spectral as spy

path = '../result/SA_label_test_predict.csv'
data = pd.read_csv(path)
data = data[['predict label']]
data = data.values
print(type(data))
gt = loadmat('../dataset/7_Salinas/Salinas_gt.mat')['salinas_gt']
view = spy.imshow(data=gt, title="salinas gt")  # 地物类别显示
# # 将预测的结果匹配到图像中
new_show = np.zeros((gt.shape[0], gt.shape[1]))
k = 0
for i in range(gt.shape[0]):
    for j in range(gt.shape[1]):
        if gt[i][j] != 0:
            new_show[i][j] = data[k]
            new_show[i][j] += 1
            k += 1

print(new_show.shape)

print(type(new_show))