import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.io as sciio
from time import time
import sklearn
# %matplotlib inline
from sklearn.datasets import make_blobs
from sklearn.decomposition import NMF
from sklearn.svm import SVC
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import skimage
from sklearn import svm, metrics, datasets
from sklearn.svm import SVC
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

from skimage.io import imread
from skimage.transform import resize
from skimage.transform import resize as imresize


# Load images in structured directory like it's sklearn sample dataset
def load_image_files(container_path, dimension=(64, 64)):  # 调整图片的尺寸为dimension=(64, 64)

    image_dir = Path(container_path)
    # folders is the list of folders each conains a category of data
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    # _______________________________________________________
    folders = folders[:-1]
    # _______________________________________________________
    categories = [fo.name for fo in folders]

    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            # print(file)
            mat_data = sciio.loadmat(file)
            img = np.abs(mat_data['frame_Ev'])
            raw_data = []
            raw_data.extend(np.real(mat_data['frame_Ev']).flatten())
            raw_data.extend(np.imag(mat_data['frame_Ev']).flatten())
            raw_data.extend(np.real(mat_data['frame_Eh']).flatten())
            raw_data.extend(np.imag(mat_data['frame_Eh']).flatten())
            # print(np.shape(raw_data))
            flat_data.append(raw_data)
            images.append(img)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)
    print("fucked")

    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)


# ----KC501-----
# image_dataset = load_image_files('/home/kc501/LJY/Alexnet/dataRCS')

# ----ICraft----
t_load_data = time()
image_dataset = load_image_files('.\dataRCS')
print("Data load done in %0.3fs" % (time() - t_load_data))
# image_dataset_test = load_image_files("E:/RL_code/alex-net-image-classification-master/class3/val")
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    image_dataset.data, image_dataset.target, test_size=0.2, random_state=101)

# X_train = image_dataset.data
# y_train = image_dataset.target
# X_test = image_dataset_test.data
# y_test = image_dataset_test.target
f = open('./TZB.txt', 'w')
for n_components in range(200, 201):
    t_NMF = time()
    nmf = NMF(n_components=n_components, init='nndsvd', tol=5e-3, max_iter=1000).fit(X_train)
    # W = nmf.components_.reshape((n_components, 64, 64))

    X_train_nmf = nmf.transform(X_train)
    X_test_nmf = nmf.transform(X_test)
    print("NMF done in %0.3fs" % (time() - t_NMF))
    # X = X_train_nmf
    # y = y_train

    # 不设定class_weight
    # clf = SVC(kernel = 'linear').fit(X,y)
    # 设定class_weight
    # c=np.arange(10,30)  #25

    t_train = time()
    clf = SVC(C=100, kernel='poly', gamma=0.01, class_weight='balanced', decision_function_shape='ovo')
    print("NMF done in %0.3fs" % (time() - t_train))
    clf = clf.fit(X_train_nmf, y_train)

    y_true_train = y_train
    y_pred_train = clf.predict(X_train_nmf)
    y_true_test = y_test
    y_pred_test = clf.predict(X_test_nmf)
    f.write("n_components: {}\n".format(n_components))
    f.write("Acc on train data: {}\n".format(accuracy_score(y_true_train, y_pred_train)))
    f.write("Acc on test data: {}\n".format(accuracy_score(y_true_test, y_pred_test)))
    f.write("------------------------------------------------------------------------------\n")
    print("n_components: ", n_components)
    print("Acc on train data:" + str(accuracy_score(y_true_train, y_pred_train)))

    print("Acc on test data:" + str(accuracy_score(y_true_test, y_pred_test)))

# recall_list=[]
# recall_0 = (y[y == clf.predict(X)] == 0).sum() / (y == 0).sum()
# recall_1 = (y[y == clf.predict(X)] == 1).sum() / (y == 1).sum()
# recall_2 = (y[y == clf.predict(X)] == 2).sum() / (y == 2).sum()
# recall_3 = (y[y == clf.predict(X)] == 3).sum() / (y == 3).sum()
# recall_4 = (y[y == clf.predict(X)] == 4).sum() / (y == 4).sum()

# print("recall_0: " + str(recall_0))
# print("recall_1: " + str(recall_1))
# print("recall_2: " + str(recall_2))
# print("recall_3: " + str(recall_3))
# print("recall_4: " + str(recall_4))

# g_means = pow(np.array(recall_0) * np.array(recall_1) * np.array(recall_2),0.5)
# print("train_g_means: " + str(g_means))

# y_pred = clf.predict(X_test_nmf)

# test_recall_0 = (y_test[y_test == clf.predict(X_test_nmf)] == 0).sum() / (y_test == 0).sum()
# test_recall_1 = (y_test[y_test == clf.predict(X_test_nmf)] == 1).sum() / (y_test == 1).sum()
# test_recall_2 = (y_test[y_test == clf.predict(X_test_nmf)] == 2).sum() / (y_test == 2).sum()
# test_recall_3 = (y_test[y_test == clf.predict(X_test_nmf)] == 3).sum() / (y_test == 3).sum()
# test_recall_4 = (y_test[y_test == clf.predict(X_test_nmf)] == 4).sum() / (y_test == 4).sum()
# print("test_recall_0"+str(test_recall_0))
# print("test_recall_1"+str(test_recall_1))
# print("test_recall_2"+str(test_recall_2))
# print("test_recall_3"+str(test_recall_3))
# print("test_recall_4"+str(test_recall_4))

# test_g_means = pow(np.array(test_recall_0) * np.array(test_recall_1) * np.array(test_recall_2),0.5)

# target_names = ["2S1", "BTR70", "BMP2"]
# print("Classification report for - \n{}:\n{}\n".format(clf, metrics.classification_report(y_test, y_pred,target_names=target_names)))
# print("test_g_means: " + str(test_g_means))
