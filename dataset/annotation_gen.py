import os
import random
import numpy as np

# saveBasePath = 'DATA_02/annotations'
# path = 'DATA_02'
# saveBasePath_linux = '../DATA_02/annotations'
# path_linux = 'DATA_02'
saveBasePath_linux = '../dataRCS/annotations'
path_linux1 = 'dataRCS'
path_linux2 = 'DATA_02'
test_ratio = 0.1
val_ratio = 0.1
random.seed(0)
files = []
for i in range(10):
    files[i] = open('{}/t{}.txt'.format(saveBasePath_linux, i), 'w')
# ftest = open('{}/test.txt'.format(saveBasePath_linux), 'w')
# ftrain = open('{}/train.txt'.format(saveBasePath_linux), 'w')
# fval = open('{}/val.txt'.format(saveBasePath_linux), 'w')

shuffled_indices = np.random.permutation(250 * 10 * 2)
# test_ratio为测试集所占的半分比
test_set_size = int(250 * test_ratio)
val_set_size = int(250 * val_ratio)
test_indices = shuffled_indices[:test_set_size]
val_indices = shuffled_indices[test_set_size:test_set_size + val_set_size]
train_indices = shuffled_indices[test_set_size + val_set_size:]
for j in test_indices:
    path_frame = path_linux + '/' + str(i + 1) + '/' + 'frame_' + str(j + 1) + '.mat' + ' ' + str(i) + '\n'
    ftest.write(path_frame)
for j in val_indices:
    path_frame = path_linux + '/' + str(i + 1) + '/' + 'frame_' + str(j + 1) + '.mat' + ' ' + str(i) + '\n'
    fval.write(path_frame)
for j in train_indices:
    path_frame = path_linux + '/' + str(i + 1) + '/' + 'frame_' + str(j + 1) + '.mat' + ' ' + str(i) + '\n'
    ftrain.write(path_frame)
ftest.close()
fval.close()
ftrain.close()
