import sys
import h5py
import numpy as np
from utils import make_print_to_file

def main():
    f=h5py.File('../h5dataset_ica_bandselect_200/BO_28_28_100.h5','r')
    data1=f['data'][:]
    f.close()
    print("BO_28_28_100:", data1.shape) 
    # (2200, 78400) 
    # 14 - 3 = 11类，超过200个样本

    f=h5py.File('../h5dataset_ica_bandselect_200/HS_28_28_100.h5','r')
    data2=f['data'][:]
    f.close()
    print("HS_28_28_100:", data2.shape) 
    # (3000, 78400) 
    # 15 - 0 = 15类，超过200个样本

    f=h5py.File('../h5dataset_ica_bandselect_200/KSC_28_28_100.h5','r')
    data3=f['data'][:]
    f.close()
    print("KSC_28_28_100:", data3.shape) 
    # (2200, 78400) 
    # 13 - 2 = 11类，超过200个样本


    f=h5py.File('../h5dataset_ica_bandselect_200/CH_28_28_100.h5','r')
    data4=f['data'][:]
    f.close()
    print("CH_28_28_100:", data4.shape)
    # (3600, 78400)
    # 19 - 1 = 18类，超过200个样本

    data=np.vstack((data1,data2,data3,data4))
    print('meta_train:', data.shape)
    # (11000, 78400)

    f=h5py.File('../h5dataset_ica_bandselect_200/meta_train_' + str(data.shape[0]) + '_' + str(data.shape[1]) + '.h5','w')
    f['data']=data
    f.close()
    print("success")

if __name__ == '__main__':
    make_print_to_file(path='./log', current_filename=sys.argv[0])
    main()