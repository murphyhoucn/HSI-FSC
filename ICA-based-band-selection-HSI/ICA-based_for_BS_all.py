'''
This code is an algorithm for band selection of hyperspectral images
using ICA (independent component analysis) method.
Reference: "Band Selection Using Independent Component Analysis for Hyperspectral
Image Processing", Hongtao Du, et al. 2003

Inputs:
        Should be given:
            A hyperspectral data-set in ENVI format
        Will be asked via the console:
            Number of components
            Number of the bands required to be selected


The number of component is determined by the user
It can be evaluated to minimise the difference
between the original band set (X) and multiplication of mixing matrix (A_) and the source (S_)

This evaluation can be checked by the 'assert' function and varying the 'atol' and 'rtol' parameters (line 62)
The smaller values of the parameters that pass the 'assert' test function, give the better estimation of the mixing matrix
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA, PCA

# to read ENVI format images
import spectral.io.envi as envi
import scipy.io
import mat73
import h5py

###########################################################


def ICA(X, n_c, n_b):
    # Whitening should be done for pre-processing of the data
    ica=FastICA(n_components=int(n_c), whiten=True)     #Indian Pine with 160 components has the best estimation of the mixing matrix
    S_ = ica.fit_transform(X)        # Reconstruct signals
    A_=ica.mixing_                 # Get estimated mixing matrix

    # To check the unmixing matrix, we can use the following line
        #assert np.allclose(X, np.dot(S_,A_.T) + ica.mean_,atol=0.0001,rtol=0.13)

    # compute the rank of a matrix for non-square matrix
    if A_.shape[1] != np.linalg.matrix_rank(A_):
        print('A does not have left inverse ')
    else:
        # compute the pseudo-inverse of A_
        W=np.linalg.pinv(A_)    # W. transpose(X)=transpose(S_)
        assert np.allclose(A_, np.dot(A_, np.dot(W, A_)))  # to check the pseudo-inverse matrix
    B_W=np.sum(np.absolute(W),axis=0)   # compute a weight per band
    sortB_W =np.argsort(B_W)   # extract the indexes

    # print('\033[94m'+'Band number with the highest scores : \n'+str(sortB_W[-int(n_b):]+1)+'\033[0m')    # to get the last elements of the list having higher weight

    # print(str(sortB_W[-int(n_b):]+1))

    ret = sortB_W[-int(n_b):]+1
    ret.sort()
    print(','.join(str(i) for i in ret))

if __name__ == '__main__':

    n_c = 1
    n_b = 100

    print("Houston")
    img = scipy.io.loadmat('../dataset/1_Houston/Houston.mat')['Houston']
    print(img.shape)
    X=np.reshape(img, (img.shape[0]*img.shape[1],img.shape[2]))
    print(X.shape)
    del img
    ICA(X, n_c, n_b)
    print("===================================================")

    print("Botswana")
    img = scipy.io.loadmat('../dataset/2_Botswana/Botswana.mat')['Botswana']
    print(img.shape)
    X=np.reshape(img, (img.shape[0]*img.shape[1],img.shape[2]))
    print(X.shape)
    del img
    ICA(X, n_c, n_b)
    print("===================================================")

    print("KSC")
    img = scipy.io.loadmat('../dataset/3_Kennedy_Space_Center/KSC.mat')['KSC']
    print(img.shape)
    X=np.reshape(img, (img.shape[0]*img.shape[1],img.shape[2]))
    print(X.shape)
    del img
    ICA(X, n_c, n_b)
    print("===================================================")

    print("chikusei")
    img = h5py.File('../dataset/4_Chikusei/Chikusei_MATLAB/HyperspecVNIR_Chikusei_20140729.mat')['chikusei']
    img = np.array(img).transpose((2, 1, 0))
    print(img.shape)
    X=np.reshape(img, (img.shape[0]*img.shape[1],img.shape[2]))
    print(X.shape)
    del img
    ICA(X, n_c, n_b)
    print("===================================================")

    print("PaviaU")
    img = scipy.io.loadmat('../dataset/5_University_of_Pavia/PaviaU.mat')['paviaU']
    print(img.shape)
    X=np.reshape(img, (img.shape[0]*img.shape[1],img.shape[2]))
    print(X.shape)
    del img
    ICA(X, n_c, n_b)
    print("===================================================")

    print("Pavia_Center")
    img = scipy.io.loadmat('../dataset/6_Pavia_Center/Pavia.mat')['pavia']
    print(img.shape)
    X=np.reshape(img, (img.shape[0]*img.shape[1],img.shape[2]))
    print(X.shape)
    del img
    ICA(X, n_c, n_b)
    print("===================================================")

    print("Salinas")
    img = scipy.io.loadmat('../dataset/7_Salinas/Salinas.mat')['salinas']
    print(img.shape)
    X=np.reshape(img, (img.shape[0]*img.shape[1],img.shape[2]))
    print(X.shape)
    del img
    ICA(X, n_c, n_b)
    print("===================================================")

    print("Indian_pines_corrected")
    img = scipy.io.loadmat('../dataset/8_Indian_Pines/Indian_pines_corrected.mat')['indian_pines_corrected']
    print(img.shape)
    X=np.reshape(img, (img.shape[0]*img.shape[1],img.shape[2]))
    print(X.shape)
    del img
    ICA(X, n_c, n_b)
    print("===================================================")

    print("xuzhou")
    img = scipy.io.loadmat('../dataset/13_Xuzhou/xuzhou.mat')['xuzhou']
    print(img.shape)
    X=np.reshape(img, (img.shape[0]*img.shape[1],img.shape[2]))
    print(X.shape)
    del img
    ICA(X, n_c, n_b)
    print("===================================================")

    print("AVIRIS-1")
    img = scipy.io.loadmat('../dataset/14_AVIRIS/AVIRIS-I.mat')['data']
    print(img.shape)
    X=np.reshape(img, (img.shape[0]*img.shape[1],img.shape[2]))
    print(X.shape)
    del img
    ICA(X, n_c, n_b)
    print("===================================================")

    print("AVIRIS-2")
    img = scipy.io.loadmat('../dataset/14_AVIRIS/AVIRIS-II.mat')['data']
    print(img.shape)
    X=np.reshape(img, (img.shape[0]*img.shape[1],img.shape[2]))
    print(X.shape)
    del img
    ICA(X, n_c, n_b)
    print("===================================================")

    print("XiongAn")
    img = mat73.loadmat('../../dataset/12_Xiongan/xiongan.mat')['XiongAn']
    print(img.shape)
    X=np.reshape(img, (img.shape[0]*img.shape[1],img.shape[2]))
    print(X.shape)
    del img
    ICA(X, n_c, n_b)
    print("===================================================")


    print("WashingtonDC")
    img = scipy.io.loadmat('../../dataset/10_WashingtonDC/dc.mat')['imggt']
    input_image = np.array(img).transpose((1, 2, 0))
    print(img.shape)
    X=np.reshape(img, (img.shape[0]*img.shape[1],img.shape[2]))
    print(X.shape)
    del img
    ICA(X, n_c, n_b)
    print("===================================================")