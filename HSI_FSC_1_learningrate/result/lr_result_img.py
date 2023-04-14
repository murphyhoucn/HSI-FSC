import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from openpyxl import load_workbook




def acc(excel_data):
    # print(excel_data.head())
    # print(excel_data)
    # print(type(excel_data))
    num = excel_data['num'].values
    lr01_data = excel_data['lr = 0.1'].values
    lr001_data = excel_data['lr = 0.01'].values
    lr0001_data = excel_data['lr = 0.001'].values
    lr00001_data = excel_data['lr = 0.0001'].values

    # print(lr01_data)
    # print(lr001_data)
    # print(lr0001_data)
    # print(lr00001_data)

    print(lr01_data.shape)
    print(type(lr01_data))

    list_num = num.tolist()
    list_lr01_data = lr01_data.tolist()
    list_lr001_data = lr001_data.tolist()
    list_lr0001_data = lr0001_data.tolist()
    list_lr00001_data = lr00001_data.tolist()
    print(len(list_lr01_data))
    print(type(list_lr01_data))

    plt.plot(list_num,list_lr01_data,':',color = 'b',label="lr = 0.1")
    plt.plot(list_num,list_lr001_data,':',color = 'c',label="lr = 0.01")
    plt.plot(list_num,list_lr0001_data,'-',color = 'r',label="lr = 0.001")
    plt.plot(list_num,list_lr00001_data,':',color = 'g',label="lr = 0.0001")
    plt.xlabel("episode")#横坐标名字
    plt.ylabel("accuracy")#纵坐标名字
    plt.legend(loc = "best")#图例
    plt.show()

def loss(excel_data):
    # print(excel_data.head())
    # print(excel_data)
    # print(type(excel_data))
    num = excel_data['num'].values
    lr01_data = excel_data['lr = 0.1'].values
    lr001_data = excel_data['lr = 0.01'].values
    lr0001_data = excel_data['lr = 0.001'].values
    lr00001_data = excel_data['lr = 0.0001'].values

    # print(lr01_data)
    # print(lr001_data)
    # print(lr0001_data)
    # print(lr00001_data)

    print(lr01_data.shape)
    print(type(lr01_data))

    list_num = num.tolist()
    list_lr01_data = lr01_data.tolist()
    list_lr001_data = lr001_data.tolist()
    list_lr0001_data = lr0001_data.tolist()
    list_lr00001_data = lr00001_data.tolist()
    print(len(list_lr01_data))
    print(type(list_lr01_data))

    plt.plot(list_num,list_lr01_data,':',color = 'b',label="lr = 0.1")
    plt.plot(list_num,list_lr001_data,':',color = 'c',label="lr = 0.01")
    plt.plot(list_num,list_lr0001_data,'-',color = 'r',label="lr = 0.001")
    plt.plot(list_num,list_lr00001_data,':',color = 'g',label="lr = 0.0001")
    
    plt.xlabel("episode")#横坐标名字
    plt.ylabel("loss")#纵坐标名字
    plt.legend(loc = "best")#图例
    plt.show()


if __name__ == '__main__':
    excel_data = pd.read_excel('./learningrate_acc_loss.xlsx', sheet_name='acc', header=0)
    acc(excel_data)
    excel_data = pd.read_excel('./learningrate_acc_loss.xlsx', sheet_name='loss', header=0)
    loss(excel_data)  