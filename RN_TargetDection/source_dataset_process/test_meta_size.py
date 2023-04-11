
import numpy as np
import h5py
from scipy.io import loadmat

#HDF5的读取
f = h5py.File('../../h5dataset_ica_bandselect_200/meta_train_10866_78400.h5','r')   #打开h5文件  # 可以查看所有的主键  
# print(type(f))

print([key for key in f.keys()])

print('first, we get values of x:', f['data'][:])
print(f['data'][:].shape)

# (10866, 78400) # 对了