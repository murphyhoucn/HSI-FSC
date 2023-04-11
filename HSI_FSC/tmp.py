data = [[]] # 一个二维数组，大小是54129*78400,里面哟很多的数值
label = [0, 0, 1,3,5,7,1,8,10......] # 一个一维的数值，大小是54129，值域是[0,15]。就是说，

data_s = []
label_s = []

for i in range(16): 
    one_class_sample_count = 0
    for j in range(54129): # 数量循环 54129:SA
        if label[j] == i and one_class_sample_count <= 4:
            data_s.append(data[j, :])
            label_s.append(label[j])
            one_class_sample_count += 1  