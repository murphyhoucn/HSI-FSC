import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from openpyxl import load_workbook




def acc_loss(excel_data):
    # print(excel_data.head())
    # print(excel_data)
    # print(type(excel_data))
    num = excel_data['num'].values
    acc = excel_data['acc'].values
    loss = excel_data['loss'].values

    list_num = num.tolist()
    list_acc_data = acc.tolist()
    list_loss_data = loss.tolist()

    plt.plot(list_num,list_acc_data,'-',color = 'b',label="accuracy")
    plt.plot(list_num,list_loss_data,'-',color = 'r',label="loss")
    plt.xlabel("episode")#横坐标名字
    plt.ylabel("loss and accuracy")#纵坐标名字
    plt.legend(loc = "best")#图例
    plt.show()


if __name__ == '__main__':
    excel_data = pd.read_excel('./NoBN_acc_loss.xlsx', sheet_name='Sheet1', header=0)
    acc_loss(excel_data)