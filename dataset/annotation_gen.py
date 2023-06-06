import os
import random
import numpy as np

# saveBasePath = 'dataRCS_2/annotations'
# path = 'dataRCS_2'
# saveBasePath_linux = '../dataRCS_2/annotations'
# path_linux = 'dataRCS_2'
saveBasePath_linux = '../openset_annotations/annotations'
path_linux1 = 'dataRCS'
path_linux2 = 'dataRCS_2'
random.seed(0)
ftest = [None] * 10
ftrain = [None] * 10
fval = [None] * 10
foval = [None] * 10
fotest = [None] * 10
strip = int(250 * 2 * 0.1)
Data = [None] * 10
for i in range(10):
    Data[i] = np.random.permutation(500)

for i in range(10):
    folder = os.path.exists(saveBasePath_linux + str(i))
    if not folder:
        os.makedirs(saveBasePath_linux + str(i))
    ftest[i] = open('{}{}/test.txt'.format(saveBasePath_linux, i), 'w')
    ftrain[i] = open('{}{}/train.txt'.format(saveBasePath_linux, i), 'w')
    fval[i] = open('{}{}/val.txt'.format(saveBasePath_linux, i), 'w')
    foval[i] = open('{}{}/open_val.txt'.format(saveBasePath_linux, i), 'w')
    fotest[i] = open('{}{}/open_test.txt'.format(saveBasePath_linux, i), 'w')


def create_path(num, x):
    i = num // 500
    y = i
    if i > x:
        y -= 1
    elif i == x:
        y = 9
    j = num % 500
    if j // 250 == 1:
        path = path_linux2
    else:
        path = path_linux1
    j %= 250
    path += '/' + str(i + 1) + '/' + 'frame_' + str(j + 1) + '.mat' + ' ' + str(y) + '\n'
    return path


for i in range(10):
    openset = Data[i]
    open_val_indices = np.array(i * 500 + openset[:int(500 * 0.1)], dtype='int')
    open_test_indices = np.array(i * 500 + openset[int(500 * 0.1):int(500 * 0.2)], dtype='int')
    val_indices = np.array([], dtype='int')
    train_indices = np.array([], dtype='int')
    test_indices = np.array([], dtype='int')

    for j in range(10):
        if i == j:
            continue
        val_indices = np.append(val_indices, j * 500 + Data[j][:int(500 * 0.1)])
        test_indices = np.append(test_indices, j * 500 + Data[j][int(500 * 0.1):int(500 * 0.2)])
        train_indices = np.append(train_indices, j * 500 + Data[j][int(500 * 0.2):])
    open_val_indices = np.append(open_val_indices, val_indices)
    open_test_indices = np.append(open_test_indices, test_indices)
    for j in test_indices:
        ftest[i].write(create_path(j, i))
    for j in val_indices:
        fval[i].write(create_path(j, i))
    for j in train_indices:
        ftrain[i].write(create_path(j, i))
    for j in open_val_indices:
        foval[i].write(create_path(j, i))
    for j in open_test_indices:
        fotest[i].write(create_path(j, i))

for i in range(10):
    ftest[i].close()
    fval[i].close()
    ftrain[i].close()
    fotest[i].close()
    foval[i].close()
