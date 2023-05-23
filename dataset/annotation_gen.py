import os
import random
import numpy as np


# saveBasePath = 'D:\TZB\Alexnet\dataRCS/annotations'
# path = 'D:\TZB\Alexnet\dataRCS'
saveBasePath_linux = '/home/kaliy/Desktop/yjy/dataRCS/annotations'
path_linux_1 = '/home/kaliy/Desktop/yjy/dataRCS'
path_linux_2 = '/home/kaliy/Desktop/yjy/dataRCS_2'
test_ratio = 0.1
val_ratio = 0.1
random.seed(0)
ftest       = open(os.path.join(saveBasePath_linux,'test.txt'), 'w')
ftrain      = open(os.path.join(saveBasePath_linux,'train.txt'), 'w')
fval        = open(os.path.join(saveBasePath_linux,'val.txt'), 'w')
for i in range(10):    # 训练集：验证集：测试集=8:1:1
    shuffled_indices1 = np.random.permutation(250)
    shuffled_indices2 = np.random.permutation(250)
    # test_ratio为测试集所占的半分比
    test_set_size = int(250 * test_ratio)
    val_set_size = int(250 * val_ratio)
    test_indices1 = shuffled_indices1[:test_set_size]
    val_indices1 = shuffled_indices1[test_set_size:test_set_size+val_set_size]
    train_indices1 = shuffled_indices1[test_set_size+val_set_size:]
    test_indices2 = shuffled_indices2[:test_set_size]
    val_indices2 = shuffled_indices2[test_set_size:test_set_size+val_set_size]
    train_indices2 = shuffled_indices2[test_set_size+val_set_size:]
    for j in test_indices1:
        path_frame = path_linux_1+'/'+str(i+1)+'/'+'frame_'+str(j+1)+'.mat'+' '+str(i)+'\n'
        ftest.write(path_frame)
    for j in val_indices1:
        path_frame = path_linux_1 + '/' + str(i+1) + '/' + 'frame_' + str(j+1) + '.mat'+' '+str(i)+'\n'
        fval.write(path_frame)
    for j in train_indices1:
        path_frame = path_linux_1 + '/' + str(i+1) + '/' + 'frame_' + str(j+1) + '.mat'+' '+str(i)+'\n'
        ftrain.write(path_frame)
    for j in test_indices2:
        path_frame = path_linux_2+'/'+str(i+1)+'/'+'frame_'+str(j+1)+'.mat'+' '+str(i)+'\n'
        ftest.write(path_frame)
    for j in val_indices2:
        path_frame = path_linux_2 + '/' + str(i+1) + '/' + 'frame_' + str(j+1) + '.mat'+' '+str(i)+'\n'
        fval.write(path_frame)
    for j in train_indices2:
        path_frame = path_linux_2 + '/' + str(i+1) + '/' + 'frame_' + str(j+1) + '.mat'+' '+str(i)+'\n'
        ftrain.write(path_frame)
ftest.close()
fval.close()
ftrain.close()





