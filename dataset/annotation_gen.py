import os
import random
import numpy as np

# saveBasePath = 'dataRCS_2/annotations'
# path = 'dataRCS_2'
# saveBasePath_linux = '../dataRCS_2/annotations'
# path_linux = 'dataRCS_2'
saveBasePath_linux = '../dataRCS/annotations'
path_linux1 = 'dataRCS'
path_linux2 = 'dataRCS_2'
random.seed(0)
ftest = [None] * 9
ftrain = [None] * 9
fval = [None] * 9
DATA = [np.array([], dtype='int')] * 10
strip = int(250 * 2 * 0.1)
for i in range(10):
    shuffled_indices = (np.random.permutation(250 * 2) + i * 250 * 2)

    for j in range(10):
        DATA[j] = np.append(DATA[j], shuffled_indices[j * strip:(j + 1) * strip])

for i in range(9):
    folder = os.path.exists(saveBasePath_linux + str(i))
    if not folder:
        os.makedirs(saveBasePath_linux + str(i))
    ftest[i] = open('{}{}/test.txt'.format(saveBasePath_linux, i), 'w')
    ftrain[i] = open('{}{}/train.txt'.format(saveBasePath_linux, i), 'w')
    fval[i] = open('{}{}/val.txt'.format(saveBasePath_linux, i), 'w')


def create_path(num):
    i = num // 500
    j = num % 500
    if j // 250 == 1:
        path = path_linux2
    else:
        path = path_linux1
    j %= 250
    path += '/' + str(i + 1) + '/' + 'frame_' + str(j + 1) + '.mat' + ' ' + str(i) + '\n'
    return path


test_indices = DATA[9]
for i in range(9):

    val_indices = DATA[i]
    train_indices = np.array([], dtype='int')
    for id, j in enumerate(DATA):
        if id == i or id == 9:
            continue
        train_indices = np.append(train_indices, j)
    for j in test_indices:
        ftest[i].write(create_path(j))
    for j in val_indices:
        fval[i].write(create_path(j))
    for j in train_indices:
        ftrain[i].write(create_path(j))

for i in range(9):
    ftest[i].close()
    fval[i].close()
    ftrain[i].close()
