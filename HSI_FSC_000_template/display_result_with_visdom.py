import pandas as pd
import numpy as np
from scipy.io import loadmat
import visdom
import seaborn as sns
import os
from utils import get_dataset_path, convert_to_color, display_predictions, load_HSI

viz = visdom.Visdom(env = 'HSI FSC Predict display')
if not viz.check_connection:
    print("Visdom is not connected. Did you run 'python -m visdom.server' ?")

if __name__ == '__main__':
    dir_path = './result/'
    file_array = os.listdir(dir_path)
    # print(file_array)
    csv_array = [i for i in file_array if i[-4:] == '.csv']
    # print(csv_array)
    for i in range(len(csv_array)):
        csv_path = dir_path + csv_array[i]
        # print(csv_path)
        data = pd.read_csv(csv_path)
        data = data[['predict label']]
        data = data.values
        # print(type(data), data.shape)
        DATASETNEME = csv_array[i][0:2]
        # print(DATASETNEME)

        img_path, gt_path, img_name, gt_name = get_dataset_path(DATASETNEME)
        img, gt = load_HSI(DATASETNEME, img_path, gt_path, img_name, gt_name)
        N_CLASSES = gt.max()
        # print(type(gt))
        # print(gt.shape)
        # print(gt.dtype)
        # np.set_printoptions(threshold=np.inf)
        # print(gt)
        # print(gt.max())

        # # 将预测的结果匹配到图像中
        new_show = np.zeros((gt.shape[0], gt.shape[1]))
        k = 0
        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                if gt[i][j] != 0:
                    new_show[i][j] = data[k]
                    new_show[i][j] += 1
                    k += 1

        # print(new_show.shape)
        # print(type(new_show))
        # print(new_show.dtype)
        new_show = (new_show).round().astype(np.uint8)
        # print(new_show.dtype)
        # print(new_show)
        # print(new_show.max())
        prediction = new_show
        palette = {0: (0, 0, 0)}
        for k, color in enumerate(sns.color_palette("hls", N_CLASSES)):
            palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype="uint8"))
        invert_palette = {v: k for k, v in palette.items()}

        color_prediction = convert_to_color(prediction, palette)

        display_predictions(
            color_prediction,
            viz,
            gt=convert_to_color(gt, palette),
            caption="Prediction vs. Ground truth",
        )