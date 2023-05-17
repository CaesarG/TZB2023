import os
import random
import numpy as np


saveBasePath = 'D:\TZB\Alexnet\dataRCS/annotations'
path = 'D:\TZB\Alexnet\dataRCS'
# saveBasePath_linux = '/home/kc501/LJY/Alexnet/dataRCS/annotations'
# path_linux = '/home/kc501/LJY/Alexnet/dataRCS'
test_ratio = 0.1
val_ratio = 0.1
random.seed(0)
ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')
ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')
fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')
for i in range(10):    # 训练集：验证集：测试集=8:1:1
    shuffled_indices = np.random.permutation(250)
    # test_ratio为测试集所占的半分比
    test_set_size = int(250 * test_ratio)
    val_set_size = int(250 * val_ratio)
    test_indices = shuffled_indices[:test_set_size]
    val_indices = shuffled_indices[test_set_size:test_set_size+val_set_size]
    train_indices = shuffled_indices[test_set_size+val_set_size:]
    for j in test_indices:
        path_frame = path+'\\'+str(i+1)+'\\'+'frame_'+str(j+1)+'.mat'+' '+str(i)+'\n'
        ftest.write(path_frame)
    for j in val_indices:
        path_frame = path + '\\' + str(i+1) + '\\' + 'frame_' + str(j+1) + '.mat'+' '+str(i)+'\n'
        fval.write(path_frame)
    for j in train_indices:
        path_frame = path + '\\' + str(i+1) + '\\' + 'frame_' + str(j+1) + '.mat'+' '+str(i)+'\n'
        ftrain.write(path_frame)
ftest.close()
fval.close()
ftrain.close()





