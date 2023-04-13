``` bash
(dlpy310pth112) PS D:\Document\DevelopProject\Develop_DeepLearning\HSI\HSI-FSC\DeepHyperX> python main.py -h                                             
usage: main.py [-h] [--dataset {PaviaC,Salinas,PaviaU,KSC,IndianPines,Botswana,Xuzhou,DFC2018_HSI}] [--model MODEL] [--folder FOLDER] [--cuda CUDA] [--runs RUNS]
               [--restore RESTORE] [--training_sample TRAINING_SAMPLE] [--sampling_mode SAMPLING_MODE] [--train_set TRAIN_SET] [--test_set TEST_SET]
               [--epoch EPOCH] [--patch_size PATCH_SIZE] [--lr LR] [--class_balancing] [--batch_size BATCH_SIZE] [--test_stride TEST_STRIDE] [--flip_augmentation]  
               [--radiation_augmentation] [--mixture_augmentation] [--with_exploration]
               [--download {PaviaC,Salinas,PaviaU,KSC,IndianPines,Botswana,Xuzhou,DFC2018_HSI} [{PaviaC,Salinas,PaviaU,KSC,IndianPines,Botswana,Xuzhou,DFC2018_HSI} ...]]

Run deep learning experiments on various hyperspectral datasets

options:
  -h, --help            show this help message and exit
  --dataset {PaviaC,Salinas,PaviaU,KSC,IndianPines,Botswana,Xuzhou,DFC2018_HSI}
                        Dataset to use.
  --model MODEL         Model to train. Available: SVM (linear), SVM_grid (grid search on linear, poly and RBF kernels), baseline (fully connected NN), hu (1D CNN), hamida (3D CNN + 1D classifier), lee (3D FCN), chen (3D CNN), li (3D CNN), he (3D CNN), luo (3D CNN), sharma (2D CNN), boulch (1D     
                        semi-supervised CNN), liu (3D semi-supervised CNN), mou (1D RNN)
  --folder FOLDER       Folder where to store the datasets (defaults to the current working directory).
  --cuda CUDA           Specify CUDA device (defaults to -1, which learns on CPU)
  --runs RUNS           Number of runs (default: 1)
  --restore RESTORE     Weights to use for initialization, e.g. a checkpoint
  --with_exploration    See data exploration visualization
  --download {PaviaC,Salinas,PaviaU,KSC,IndianPines,Botswana,Xuzhou,DFC2018_HSI} [{PaviaC,Salinas,PaviaU,KSC,IndianPines,Botswana,Xuzhou,DFC2018_HSI} ...]
                        Download the specified datasets and quits.

Dataset:
  --training_sample TRAINING_SAMPLE
                        Percentage of samples to use for training (default: 10%)
  --sampling_mode SAMPLING_MODE
                        Sampling mode (random sampling or disjoint, default: random)
  --train_set TRAIN_SET
                        Path to the train ground truth (optional, this supersedes the --sampling_mode option)
  --test_set TEST_SET   Path to the test set (optional, by default the test_set is the entire ground truth minus the training)

Training:
  --epoch EPOCH         Training epochs (optional, if absent will be set by the model)
  --patch_size PATCH_SIZE
                        Size of the spatial neighbourhood (optional, if absent will be set by the model)
  --lr LR               Learning rate, set by the model if not specified.
  --class_balancing     Inverse median frequency class balancing (default = False)
  --batch_size BATCH_SIZE
                        Batch size (optional, if absent will be set by the model
  --test_stride TEST_STRIDE
                        Sliding window step stride during inference (default = 1)

Data augmentation:
  --flip_augmentation   Random flips (if patch_size > 1)
  --radiation_augmentation
                        Random radiation noise (illumination)
  --mixture_augmentation
                        Random mixes between spectra
(dlpy310pth112) PS D:\Document\DevelopProject\Develop_DeepLearning\HSI\HSI-FSC\DeepHyperX> 
```





# 1. SVM

## IP

``` bash
(dlpy310pth112) PS D:\Document\DevelopProject\Develop_DeepLearning\HSI\HSI-FSC\DeepHyperX> python main.py --model SVM --dataset IndianPines --cuda 0
Computation on CUDA GPU device 0
Setting up a new session...

=====================<dataset get_dataset>Murphy 13-Apr-23===============================
(145, 145, 200)
(145, 145)
16 0
=====================<dataset get_dataset>Murphy 13-Apr-23===============================

Image has dimensions 145x145 and 200 channels

=====================<utils sample_gt> murphy_5sample_perclass Murphy 13-Apr-23===============================
[(73, 100), (73, 99), (64, 96), (68, 100), (69, 101), (65, 37), (43, 54), (66, 39), (58, 96), (90, 82), (60, 4), (8, 23), (12, 22), (62, 3), (6, 11), (36, 9), (49, 9), (45, 22), (44, 15), (42, 7), (7, 26), (76, 10), (79, 17), (74, 114), (68, 114), (111, 66), (49, 29), (102, 81), (53, 26), (102, 77), (74, 110), (78, 110), (75, 111), (72, 108), (78, 111), (43, 134), (58, 135), (34, 126), (50, 139), (48, 133), (70, 23), (65, 23), (63, 22), (69, 22), (62, 22), (37, 82), (9, 30), (62, 90), (59, 75), (65, 81), (12, 100), (58, 37), (16, 115), (8, 115), (76, 63), (12, 36), (18, 52), (2, 29), (8, 36), (13, 39), (119, 40), (121, 35), (118, 27), (117, 33), (124, 42), (32, 102), (38, 115), (130, 100), (128, 90), (33, 98), (5, 74), (7, 82), (5, 87), (4, 76), (1, 76), (27, 50), (20, 44), (20, 48), (21, 46), (23, 43)]      
=====================<utils sample_gt> murphy_5sample_perclass Murphy 13-Apr-23===============================


===============================<main> Murphy 13-Apr-23=======================================
10169
{0: 20945, 1: 5, 2: 5, 3: 5, 4: 5, 5: 5, 6: 5, 7: 5, 8: 5, 9: 5, 10: 5, 11: 5, 12: 5, 13: 5, 14: 5, 15: 5, 16: 5}
21025
{0: 10776, 1: 46, 2: 1428, 3: 830, 4: 237, 5: 483, 6: 730, 7: 28, 8: 478, 9: 20, 10: 972, 11: 2455, 12: 593, 13: 205, 14: 1265, 15: 386, 16: 93}
21025
===============================<main> Murphy 13-Apr-23=======================================

80 samples selected (over 10249)
Running an experiment with the SVM model run 1/1
Saving model params in 2023_04_13_20_09_23

===============================<utils metrics> murphy 13-apr-23=======================================
1
16
===============================<utils metrics> murphy 13-apr-23=======================================

D:\Document\DevelopProject\Develop_DeepLearning\HSI\HSI-FSC\DeepHyperX\utils.py:406: RuntimeWarning: invalid value encountered in double_scalars
  F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
Confusion matrix :
[[   0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0]
 [   0   39    0    0    0    0    0    1    6    0    0    0    0    0
     0    0    0]
 [   0    1  430  368  259    0    5   25    0    2    0  224  112    0
     0    1    1]
 [   0    0  104  412   71    0    2    8    0    0    0  174   59    0
     0    0    0]
 [   0    0    3   25  110    0   27   38    1    2    0    3   22    6
     0    0    0]
 [   0   33    1    4    2    0    9   24    3   84    0    0    4    0
   318    1    0]
 [   0    1    0    0    0    0  510    0    0   31    0    0    0   31
    44  113    0]
 [   0    1    0    0    0    0    0   26    1    0    0    0    0    0
     0    0    0]
 [   0   93    0    0    0    0    2  154  228    1    0    0    0    0
     0    0    0]
 [   0    0    0    0    0    0    4    0    0   12    0    0    0    4
     0    0    0]
 [   0    0   68  399    0    0    3   62    0    1    0  366   73    0
     0    0    0]
 [   0    0  396 1343    1    0    8  108    0    8    0  439  142    3
     0    0    7]
 [   0    0  170   76   51    0    2   19    0    0    0  129  146    0
     0    0    0]
 [   0    0    0    0    0    0   11    1    0    3    0    0    0  190
     0    0    0]
 [   0    0    0    0    0    0    1    0    0    0    0    0    0  138
  1112   14    0]
 [   0    7    0    4    0    0   74    8    0   39    0    1    9  115
F1 scores :
        Undefined: nan
        Alfalfa: 0.353
        Corn-notill: 0.330
        Corn-mintill: 0.238
        Corn: 0.301
        Grass-pasture: 0.000
        Grass-trees: 0.735
        Grass-pasture-mowed: 0.104
        Hay-windrowed: 0.636
        Oats: 0.118
        Soybean-notill: 0.000
        Soybean-mintill: 0.232
        Soybean-clean: 0.252
        Wheat: 0.549
        Woods: 0.784
        Buildings-Grass-Trees-Drives: 0.117
        Stone-Steel-Towers: 0.890
---
Kappa: 0.298

AA = 33.16783721870909 %
(dlpy310pth112) PS D:\Document\DevelopProject\Develop_DeepLearning\HSI\HSI-FSC\DeepHyperX>


```

## SA

``` bash
(dlpy310pth112) PS D:\Document\DevelopProject\Develop_DeepLearning\HSI\HSI-FSC\DeepHyperX> python main.py --model SVM --dataset Salinas --cuda 0    
Computation on CUDA GPU device 0
Setting up a new session...

=====================<dataset get_dataset>Murphy 13-Apr-23===============================
(512, 217, 204)
(512, 217)
16 0
=====================<dataset get_dataset>Murphy 13-Apr-23===============================

Image has dimensions 512x217 and 204 channels

=====================<utils sample_gt> murphy_5sample_perclass Murphy 13-Apr-23===============================
[(245, 43), (259, 49), (251, 20), (264, 36), (247, 28), (223, 135), (212, 174), (236, 110), (235, 110), (208, 117), (179, 170), (173, 169), (167, 168), (151, 170), (170, 177), (42, 80), (132, 126), (114, 118), (18, 65), (148, 127), (125, 126), (84, 120), (26, 82), (41, 99), (94, 121), (51, 123), (85, 138), (114, 167), (63, 139), (77, 143), (56, 149), (3, 126), (1, 134), (83, 178), (51, 149), (170, 40), (152, 121), (204, 44), (177, 108), (103, 94), (414, 63), (432, 89), (396, 62), (462, 84), (413, 15), (338, 75), (333, 89), (367, 18), (336, 52), (345, 53), (276, 38), (299, 11), (276, 43), (301, 9), (264, 61), (297, 44), (280, 63), (284, 73), (325, 7), (261, 96), (324, 23), (319, 27), (281, 105), (306, 61), (292, 78), (313, 60), (310, 68), (298, 99), (318, 59), (300, 95), (81, 3), (97, 56), (38, 21), (13, 54), (111, 27), (472, 6), (478, 21), (466, 6), (454, 5), (474, 5)]
=====================<utils sample_gt> murphy_5sample_perclass Murphy 13-Apr-23===============================


===============================<main> Murphy 13-Apr-23=======================================
54049
{0: 111024, 1: 5, 2: 5, 3: 5, 4: 5, 5: 5, 6: 5, 7: 5, 8: 5, 9: 5, 10: 5, 11: 5, 12: 5, 13: 5, 14: 5, 15: 5, 16: 5}
111104
{0: 56975, 1: 2009, 2: 3726, 3: 1976, 4: 1394, 5: 2678, 6: 3959, 7: 3579, 8: 11271, 9: 6203, 10: 3278, 11: 1068, 12: 1927, 13: 916, 14: 1070, 15: 7268, 16: 1807}
111104
===============================<main> Murphy 13-Apr-23=======================================

80 samples selected (over 54129)
Running an experiment with the SVM model run 1/1
Saving model params in 2023_04_13_20_10_52

===============================<utils metrics> murphy 13-apr-23=======================================
1
16
===============================<utils metrics> murphy 13-apr-23=======================================

D:\Document\DevelopProject\Develop_DeepLearning\HSI\HSI-FSC\DeepHyperX\utils.py:406: RuntimeWarning: invalid value encountered in double_scalars
  F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
Confusion matrix :
[[    0     0     0     0     0     0     0     0     0     0     0     0
      0     0     0     0     0]
 [    0  2002     7     0     0     0     0     0     0     0     0     0
      0     0     0     0     0]
 [    0  1826  1693     0     0     0     0   178     4     0     0     0
      0     1    24     0     0]
 [    0     0     0  1405     0     4     0     0     0   128     0     4
    435     0     0     0     0]
 [    0     0     0     2  1330    62     0     0     0     0     0     0
      0     0     0     0     0]
 [    0     0     0  2606    18    43     0     0     0     2     0     4
      5     0     0     0     0]
 [    0     0     0     1     0     0  3772     0     0     0     0   165
      1     5    15     0     0]
 [    0     0     0     2     0     0     0  3552    10     0     0     0
      0     5     3     0     7]
 [    0     0     0     0     0    15     0    21 10749     2    12    81
     61   301    14     0    15]
 [    0     0     0     0     0     0     0     0     0  5314     3   880
      0     6     0     0     0]
 [    0     0     0    48     3     5     0     0    71   377    70  1695
    355   623    31     0     0]
 [    0     0     0     6     0     0     0     0     0    94     0   917
     51     0     0     0     0]
 [    0     0     0   358     0     0     0     0     0     2     0     0
   1567     0     0     0     0]
 [    0     0     0     0     0     0     0     0     0     0     0     0
      0   907     9     0     0]
 [    0     0     0     0     0     0     0     0    21     0     1     7
      0   123   918     0     0]
 [    0     0     0    24     4     9     0     4  7049    11     0     0
     70    79     3     0    15]
 [    0     0     0     6    50    29     0   446   847     0     1     0
      9     0     4     0   415]]---
Accuracy(OA) : 64.021%
---
F1 scores :
        Undefined: nan
        Brocoli_green_weeds_1: 0.686
        Brocoli_green_weeds_2: 0.624
        Fallow: 0.437
        Fallow_rough_plow: 0.950
        Fallow_smooth: 0.030
        Stubble: 0.976
        Celery: 0.913
        Grapes_untrained: 0.716
        Soil_vinyard_develop: 0.876
        Corn_senesced_green_weeds: 0.042
        Lettuce_romaine_4wk: 0.380
        Lettuce_romaine_5wk: 0.699
        Lettuce_romaine_6wk: 0.612
        Lettuce_romaine_7wk: 0.878
        Vinyard_untrained: 0.000
        Vinyard_vertical_trellis: 0.367
---
Kappa: 0.597

AA = 54.03972924806354 %
(dlpy310pth112) PS D:\Document\DevelopProject\Develop_DeepLearning\HSI\HSI-FSC\DeepHyperX> 
```

## UP

``` bash
(dlpy310pth112) PS D:\Document\DevelopProject\Develop_DeepLearning\HSI\HSI-FSC\DeepHyperX> python main.py --model SVM --cuda 0 --dataset PaviaU
Computation on CUDA GPU device 0
Setting up a new session...

=====================<dataset get_dataset>Murphy 13-Apr-23===============================
(610, 340, 103)
(610, 340)
9 0
=====================<dataset get_dataset>Murphy 13-Apr-23===============================

Image has dimensions 610x340 and 103 channels

=====================<utils sample_gt> murphy_5sample_perclass Murphy 13-Apr-23===============================
[(467, 173), (59, 130), (126, 163), (121, 162), (379, 338), (567, 166), (575, 252), (600, 45), (595, 161), (564, 227), (377, 29), (424, 6), (318, 23), (389, 21), (415, 66), (257, 12), (554, 7), (7, 168), (398, 281), (490, 163), (226, 152), (208, 149), (210, 147), (185, 148), (210, 144), (349, 168), (298, 188), (345, 178), (290, 174), (350, 204), (321, 125), (338, 122), (354, 131), (301, 161), (365, 159), (511, 123), (444, 141), (161, 53), (538, 29), (179, 61), (323, 150), (481, 34), (223, 136), (256, 160), (433, 18)]
=====================<utils sample_gt> murphy_5sample_perclass Murphy 13-Apr-23===============================


===============================<main> Murphy 13-Apr-23=======================================
42731
{0: 207355, 1: 5, 2: 5, 3: 5, 4: 5, 5: 5, 6: 5, 7: 5, 8: 5, 9: 5}
207400
{0: 164624, 1: 6631, 2: 18649, 3: 2099, 4: 3064, 5: 1345, 6: 5029, 7: 1330, 8: 3682, 9: 947}
207400
===============================<main> Murphy 13-Apr-23=======================================

45 samples selected (over 42776)
Running an experiment with the SVM model run 1/1
Saving model params in 2023_04_13_20_12_23

===============================<utils metrics> murphy 13-apr-23=======================================
1
9
===============================<utils metrics> murphy 13-apr-23=======================================

D:\Document\DevelopProject\Develop_DeepLearning\HSI\HSI-FSC\DeepHyperX\utils.py:406: RuntimeWarning: invalid value encountered in double_scalars
  F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
Confusion matrix :
[[    0     0     0     0     0     0     0     0     0     0]
 [    0  3571     0   125     0    30    39  2653   212     1]
 [    0     9 10532   504  2414     0  5186     1     3     0]
 [    0    17     0   657     0     0     5   750   670     0]
 [    0     0   670     0  2368     9    14     0     0     3]
 [    0     0     0     0     0  1325     1     3    16     0]
 [    0    17  1467   450     1    66  2602    10   416     0]
 [    0    20     0     1     0     0     0  1308     1     0]
 [    0     7     1   665     0     0    23   480  2506     0]
 [    0     1     0     0     0     0     0     0     0   946]]---
Accuracy(OA) : 60.349%
---
F1 scores :
        Undefined: nan
        Asphalt: 0.695
        Meadows: 0.673
        Gravel: 0.292
        Trees: 0.604
        Painted metal sheets: 0.955
        Bare Soil: 0.403
        Bitumen: 0.400
        Self-Blocking Bricks: 0.668
        Shadows: 0.997
---
Kappa: 0.512

AA = 56.87061209198085 %
(dlpy310pth112) PS D:\Document\DevelopProject\Develop_DeepLearning\HSI\HSI-FSC\DeepHyperX> 
```

## PC


``` bash
(dlpy310pth112) PS D:\Document\DevelopProject\Develop_DeepLearning\HSI\HSI-FSC\DeepHyperX> python main.py --model SVM --cuda 0 --dataset PaviaC
Computation on CUDA GPU device 0
Setting up a new session...

=====================<dataset get_dataset>Murphy 13-Apr-23===============================
(1096, 715, 102)
(1096, 715)
9 0
=====================<dataset get_dataset>Murphy 13-Apr-23===============================

Image has dimensions 1096x715 and 102 channels

=====================<utils sample_gt> murphy_5sample_perclass Murphy 13-Apr-23===============================
[(89, 277), (48, 281), (136, 450), (3, 183), (221, 624), (261, 699), (764, 126), (157, 275), (6, 481), (1038, 657), (454, 260), (232, 342), (233, 412), (435, 707), (327, 483), (894, 182), (385, 455), (382, 486), (430, 431), (345, 458), (287, 493), (257, 463), (343, 519), (360, 543), (290, 499), (277, 367), (328, 435), (425, 330), (141, 223), (989, 163), (537, 570), (626, 654), (535, 582), (658, 277), (595, 596), (523, 49), (447, 14), (640, 135), (560, 181), (456, 214), (461, 318), (480, 344), (475, 313), (336, 55), (397, 396)]
=====================<utils sample_gt> murphy_5sample_perclass Murphy 13-Apr-23===============================


===============================<main> Murphy 13-Apr-23=======================================
148107
{0: 783595, 1: 5, 2: 5, 3: 5, 4: 5, 5: 5, 6: 5, 7: 5, 8: 5, 9: 5}
783640
{0: 635488, 1: 65971, 2: 7598, 3: 3090, 4: 2685, 5: 6584, 6: 9248, 7: 7287, 8: 42826, 9: 2863}
783640
===============================<main> Murphy 13-Apr-23=======================================

45 samples selected (over 148152)
Running an experiment with the SVM model run 1/1
Saving model params in 2023_04_13_20_14_17

===============================<utils metrics> murphy 13-apr-23=======================================
1
9
===============================<utils metrics> murphy 13-apr-23=======================================

D:\Document\DevelopProject\Develop_DeepLearning\HSI\HSI-FSC\DeepHyperX\utils.py:406: RuntimeWarning: invalid value encountered in double_scalars
  F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
Confusion matrix :
[[    0     0     0     0     0     0     0     0     0     0]
 [    0 65491     0     0     0     0   480     0     0     0]
 [    0     0  5582  1934     0     0    76     0     0     6]
 [    0     0   283  2795     0    12     0     0     0     0]
 [    0     0     0     0  2199   225    58   196     7     0]
 [    0     0     0    55  1699  4575     1     7   247     0]
 [    0     0     0     0    72     6  9008   156     6     0]
 [    0     5     0     0   517    44  4015  2704     2     0]
 [    0    12     0     0   443   370  2811    29 39161     0]
 [    0    77     0     0     0     0     0     0     0  2786]]---
Accuracy(OA) : 90.651%
---
F1 scores :
        Undefined: nan
        Water: 0.996
        Trees: 0.829
        Asphalt: 0.710
        Self-Blocking Bricks: 0.578
        Bitumen: 0.774
        Tiles: 0.701
        Shadows: 0.521
        Meadows: 0.952
        Bare Soil: 0.985
---
Kappa: 0.869

AA = 70.464450921224 %
(dlpy310pth112) PS D:\Document\DevelopProject\Develop_DeepLearning\HSI\HSI-FSC\DeepHyperX> 
```


## XZ

``` bash
(dlpy310pth112) PS D:\Document\DevelopProject\Develop_DeepLearning\HSI\HSI-FSC\DeepHyperX> python main.py --model SVM --cuda 0 --dataset Xuzhou
Computation on CUDA GPU device 0
Setting up a new session...

=====================<dataset get_dataset>Murphy 13-Apr-23===============================
(500, 260, 436)
(500, 260)
9 0
=====================<dataset get_dataset>Murphy 13-Apr-23===============================

Image has dimensions 500x260 and 436 channels

=====================<utils sample_gt> murphy_5sample_perclass Murphy 13-Apr-23===============================
[(267, 22), (193, 82), (316, 4), (357, 27), (492, 23), (467, 108), (438, 136), (459, 159), (472, 119), (462, 116), (456, 252), (320, 237), (344, 247), (483, 241), (455, 225), (56, 249), (384, 200), (212, 224), (187, 227), (109, 177), (162, 177), (210, 195), (190, 138), (168, 144), (277, 94), (40, 219), (45, 210), (473, 95), (483, 93), (474, 86), (430, 32), (441, 21), (399, 46), (383, 34), (463, 38), (55, 29), (3, 55), (94, 15), (25, 21), (17, 22), (11, 158), (39, 161), (45, 141), (4, 114), (102, 153)]
=====================<utils sample_gt> murphy_5sample_perclass Murphy 13-Apr-23===============================


===============================<main> Murphy 13-Apr-23=======================================
68832
{0: 129955, 1: 5, 2: 5, 3: 5, 4: 5, 5: 5, 6: 5, 7: 5, 8: 5, 9: 5}
130000
{0: 61123, 1: 26396, 2: 4027, 3: 2783, 4: 5214, 5: 13184, 6: 2436, 7: 6990, 8: 4777, 9: 3070}
130000
===============================<main> Murphy 13-Apr-23=======================================

45 samples selected (over 68877)
Running an experiment with the SVM model run 1/1
Saving model params in 2023_04_13_20_15_27

===============================<utils metrics> murphy 13-apr-23=======================================
1
9
===============================<utils metrics> murphy 13-apr-23=======================================

D:\Document\DevelopProject\Develop_DeepLearning\HSI\HSI-FSC\DeepHyperX\utils.py:406: RuntimeWarning: invalid value encountered in double_scalars
  F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
Confusion matrix :
[[    0     0     0     0     0     0     0     0     0     0]
 [    0 19861     0     2  2491    63  1430  2450    99     0]
 [    0     0  3958    14     1     1    14     0    39     0]
 [    0     0  1617   970    34     0    79     0    83     0]
 [    0   502    89   522  3336     0   508     5   252     0]
 [    0   236     0     1   188  6462  4282  1432   583     0]
 [    0     0    25    73    14    19   945   277  1083     0]
 [    0   400     0     0    58    57  2857  3613     5     0]
 [    0     0    56  1136     0    44    83     0  3458     0]
 [    0   102   663   347    89     0  1310    22   524    13]]---
Accuracy(OA) : 61.873%
---
F1 scores :
        Undefined: nan
        Bareland1: 0.836
        Lakes: 0.759
        Coals: 0.332
        Cement: 0.584
        Crops-1: 0.652
        Tress: 0.136
        Bareland2: 0.489
        Crops: 0.634
        Red-title: 0.008
---
Kappa: 0.539

AA = 44.29268680778847 %
(dlpy310pth112) PS D:\Document\DevelopProject\Develop_DeepLearning\HSI\HSI-FSC\DeepHyperX> 
```


# nn

## IP

``` bash
****************main.py_2023-04-13_21_05_34*****************
Computation on CUDA GPU device 0

=====================<dataset get_dataset>Murphy 13-Apr-23===============================
(145, 145, 200)
(145, 145)
16 0
=====================<dataset get_dataset>Murphy 13-Apr-23===============================

Image has dimensions 145x145 and 200 channels

=====================<utils sample_gt> murphy_5sample_perclass Murphy 13-Apr-23===============================
[(74, 101), (69, 98), (70, 100), (66, 97), (72, 101), (89, 74), (91, 78), (41, 105), (47, 62), (34, 45), (130, 23), (134, 38), (30, 18), (2, 14), (127, 45), (43, 12), (41, 14), (45, 20), (45, 17), (49, 14), (83, 10), (79, 14), (74, 6), (74, 11), (75, 1), (104, 66), (104, 89), (109, 75), (58, 29), (64, 29), (74, 108), (76, 110), (72, 111), (77, 109), (73, 108), (51, 136), (33, 129), (48, 121), (58, 138), (49, 123), (63, 22), (67, 22), (64, 23), (66, 22), (69, 22), (59, 86), (54, 87), (53, 108), (48, 77), (53, 75), (109, 43), (79, 66), (105, 35), (16, 113), (91, 60), (17, 64), (22, 37), (24, 55), (25, 38), (54, 14), (124, 31), (122, 26), (117, 39), (117, 25), (123, 29), (9, 122), (35, 108), (120, 115), (14, 124), (122, 98), (20, 27), (3, 86), (22, 27), (8, 94), (3, 77), (15, 49), (27, 48), (16, 46), (23, 50), (24, 48)]
=====================<utils sample_gt> murphy_5sample_perclass Murphy 13-Apr-23===============================


===============================<main> Murphy 13-Apr-23=======================================
10169
{0: 20945, 1: 5, 2: 5, 3: 5, 4: 5, 5: 5, 6: 5, 7: 5, 8: 5, 9: 5, 10: 5, 11: 5, 12: 5, 13: 5, 14: 5, 15: 5, 16: 5}
21025
{0: 10776, 1: 46, 2: 1428, 3: 830, 4: 237, 5: 483, 6: 730, 7: 28, 8: 478, 9: 20, 10: 972, 11: 2455, 12: 593, 13: 205, 14: 1265, 15: 386, 16: 93}
21025
===============================<main> Murphy 13-Apr-23=======================================

80 samples selected (over 10249)
Running an experiment with the nn model run 1/1

=====================<utils sample_gt> murphy_5sample_perclass Murphy 13-Apr-23===============================
[(69, 100), (74, 100), (73, 101), (65, 97), (69, 98), (39, 58), (52, 70), (63, 44), (64, 43), (80, 80), (126, 26), (3, 8), (5, 8), (7, 6), (34, 18), (43, 19), (35, 4), (43, 8), (39, 3), (49, 15), (74, 115), (118, 51), (68, 114), (84, 1), (78, 17), (70, 34), (48, 26), (104, 89), (99, 70), (108, 69), (77, 111), (75, 110), (78, 108), (76, 109), (72, 110), (51, 128), (57, 139), (50, 134), (44, 122), (43, 124), (69, 22), (61, 22), (66, 22), (70, 22), (63, 22), (57, 81), (15, 33), (51, 89), (52, 80), (47, 74), (71, 61), (81, 66), (18, 114), (94, 40), (62, 37), (16, 43), (20, 70), (4, 33), (11, 56), (22, 40), (117, 37), (118, 29), (122, 24), (123, 34), (120, 26), (133, 116), (138, 84), (132, 117), (129, 96), (120, 92), (13, 88), (7, 77), (22, 26), (19, 32), (9, 93), (23, 42), (19, 44), (21, 47), (16, 47), (23, 50)]
=====================<utils sample_gt> murphy_5sample_perclass Murphy 13-Apr-23===============================

{'dataset': 'IndianPines', 'model': 'nn', 'folder': './Datasets/', 'cuda': 0, 'runs': 1, 'training_sample': 10, 'sampling_mode': 'murphy_5sample_perclass', 'class_balancing': False, 'test_stride': 1, 'flip_augmentation': False, 'radiation_augmentation': False, 'mixture_augmentation': False, 'with_exploration': False, 'n_classes': 17, 'n_bands': 200, 'ignored_labels': [0], 'device': device(type='cuda', index=0), 'weights': tensor([0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       device='cuda:0'), 'patch_size': 1, 'dropout': False, 'learning_rate': 0.0001, 'epoch': 100, 'batch_size': 100, 'scheduler': <torch.optim.lr_scheduler.ReduceLROnPlateau object at 0x000001CD42295390>, 'supervision': 'full', 'center_pixel': True}
Network :
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                 [-1, 2048]         411,648
            Linear-2                 [-1, 4096]       8,392,704
            Linear-3                 [-1, 2048]       8,390,656
            Linear-4                   [-1, 17]          34,833
================================================================
Total params: 17,229,841
Trainable params: 17,229,841
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.06
Params size (MB): 65.73
Estimated Total Size (MB): 65.79
----------------------------------------------------------------
Saving neural network weights in 2023_04_13_21_05_46_epoch5_0.31
Saving neural network weights in 2023_04_13_21_05_53_epoch10_0.37
Saving neural network weights in 2023_04_13_21_06_00_epoch15_0.39
Saving neural network weights in 2023_04_13_21_06_08_epoch20_0.42
Saving neural network weights in 2023_04_13_21_06_15_epoch25_0.43
Saving neural network weights in 2023_04_13_21_06_22_epoch30_0.45
Saving neural network weights in 2023_04_13_21_06_29_epoch35_0.47
Saving neural network weights in 2023_04_13_21_06_36_epoch40_0.48
Saving neural network weights in 2023_04_13_21_06_44_epoch45_0.48
Saving neural network weights in 2023_04_13_21_06_51_epoch50_0.49
Saving neural network weights in 2023_04_13_21_06_58_epoch55_0.49
Saving neural network weights in 2023_04_13_21_07_06_epoch60_0.49
Saving neural network weights in 2023_04_13_21_07_13_epoch65_0.51
Saving neural network weights in 2023_04_13_21_07_20_epoch70_0.51
Saving neural network weights in 2023_04_13_21_07_27_epoch75_0.51
Saving neural network weights in 2023_04_13_21_07_35_epoch80_0.50
Saving neural network weights in 2023_04_13_21_07_42_epoch85_0.51
Saving neural network weights in 2023_04_13_21_07_49_epoch90_0.51
Saving neural network weights in 2023_04_13_21_07_56_epoch95_0.51
Train (epoch 100/100) [0/80 (0%)]	Loss: 1.068039
Saving neural network weights in 2023_04_13_21_08_03_epoch100_0.52

===============================<utils metrics> murphy 13-apr-23=======================================
1
16
===============================<utils metrics> murphy 13-apr-23=======================================

Confusion matrix :
[[   0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0]
 [   0   36    0    0    0    0    0    1    9    0    0    0    0    0
     0    0    0]
 [   0    0  397  189  137    0    2    2    4    1  147  261  277    1
     0    7    3]
 [   0    0   83  403   32    0    0    1    0    1   55  132  114    0
     0    9    0]
 [   0    0    0   29  128    0    4    7   11    5    0    0   21    1
     0   31    0]
 [   0    6    0    0   13  235    9    7   13   17    0    0    4    0
   169   10    0]
 [   0    0    0    0    0   46  454    0   25   64    0    0    0   23
     0  118    0]
 [   0    0    0    0    0    1    0   27    0    0    0    0    0    0
     0    0    0]
 [   0   70    0    0   25    0    0  118  265    0    0    0    0    0
     0    0    0]
 [   0    0    0    0    0    0    0    0    0   15    0    0    0    5
     0    0    0]
 [   0    0   11   82   82    0    0   10    5    1  672   22   84    0
     0    3    0]
 [   0    0  253  381   92    1    2   17   12    7  451 1076  157    0
     0    6    0]
 [   0    0   98   67   15    2    0    5    3    0   78   80  220    0
     0   24    1]
 [   0    0    0    0    1    0    2    0    0   10    0    0    0  192
     0    0    0]
 [   0    0    0    0    0  135    3    0    0    0    0    0    0    2
   980  145    0]
 [   0    1    0    0    0   59   54    0    0   33    0    0    0   35
    52  149    3]
 [   0    0    7    0    0    0    0    0    0    0    2    0    0    0
     0    0   84]]---
Accuracy(OA) : 52.034%
---
F1 scores :
	Undefined: nan
	Alfalfa: 0.453
	Corn-notill: 0.349
	Corn-mintill: 0.407
	Corn: 0.336
	Grass-pasture: 0.489
	Grass-trees: 0.721
	Grass-pasture-mowed: 0.242
	Hay-windrowed: 0.642
	Oats: 0.172
	Soybean-notill: 0.565
	Soybean-mintill: 0.535
	Soybean-clean: 0.299
	Wheat: 0.828
	Woods: 0.795
	Buildings-Grass-Trees-Drives: 0.336
	Stone-Steel-Towers: 0.913
---
Kappa: 0.466

AA = 47.53433732604162 %

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```


# xxxx

## IP

``` bash

```

## SA

``` bash

```

## UP

``` bash

```

## PC


``` bash

```


## XZ

``` bash

```

