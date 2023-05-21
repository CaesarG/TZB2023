import scipy.io as sciio
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

from IPython.core.debugger import set_trace

"""
用于观察数据的直观特征
"""
path_rcs = r'D:\TZB2023\dataRCS\3\frame_131.mat'
data_rcs = sciio.loadmat(path_rcs)    # sciio加载的每帧mat数据为一个字典
data_Ev = data_rcs['frame_Ev'].astype(np.complex)
data_Eh = data_rcs['frame_Eh'].astype(np.complex)

ifft_Ev = np.abs(np.fft.ifftshift(np.fft.ifft(data_Ev.T))).astype(np.float32)
ifft_Eh = np.abs(np.fft.ifftshift(np.fft.ifft(data_Eh.T))).astype(np.float32)

rcs_ifft = np.concatenate((np.expand_dims(ifft_Ev, axis=0), np.expand_dims(ifft_Eh, axis=0)), axis=0)

del ifft_Ev
del ifft_Eh

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(0, 401, 1)
Y = np.arange(0, 512, 1)
X, Y = np.meshgrid(Y, X)
Z = rcs_ifft[0]
Z_MAX = np.max(rcs_ifft[1])
Z = (Z/Z_MAX*255).astype(np.uint8).T
Z_tensor = torch.tensor(Z)
plt.imsave('D:\TZB2023\dataRCS\observer.png', Z_tensor)
# nmb = cv2.imread('D:\TZB2023\dataRCS\observer.png', 0)

Z_hist = cv2.equalizeHist(Z)
plt.imsave('D:\TZB2023\dataRCS\eq_HIST.png', Z_hist)
nmb = cv2.imread('D:\TZB2023\dataRCS\eq_HIST.png', 0)
set_trace()

