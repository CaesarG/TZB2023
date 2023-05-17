from torch.utils.data import Dataset
import scipy.io as sciio
import numpy as np
from IPython.core.debugger import set_trace

"""
注：生成标签annotation_lines为txt格式，每一行对应一帧的数据，分别记录帧文件路径和该帧类别
self.annotation_lines为一个数组，每一个元素记录原标签txt一行的字符串。
"""
class myDataset_ifft(Dataset):
    def __init__(self, phase, annotation_lines):
        super(myDataset_ifft, self).__init__()
        with open(annotation_lines, 'r') as f:
            self.annotation_lines = f.readlines()
        self.length = len(self.annotation_lines)
        self.phase = phase

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        rcs_ifft, gt = self.get_rcs_ifft(self.annotation_lines[index], self.phase)
        return rcs_ifft, gt

    def get_rcs_ifft(self, annotation_line, phase):
        line = annotation_line.split()
        path_rcs = line[0]
        ground_truth = int(line[1])    # 类别标签值
        data_rcs = sciio.loadmat(path_rcs)    # sciio加载的每帧mat数据为一个字典
        data_Ev = data_rcs['frame_Ev'].astype(np.complex)
        data_Eh = data_rcs['frame_Eh'].astype(np.complex)

        ifft_Ev = np.abs(np.fft.ifftshift(np.fft.ifft(data_Ev.T))).astype(np.float32)
        ifft_Eh = np.abs(np.fft.ifftshift(np.fft.ifft(data_Eh.T))).astype(np.float32)

        rcs_ifft = np.concatenate((np.expand_dims(ifft_Ev.T, axis=0), np.expand_dims(ifft_Eh.T, axis=0)), axis=0)

        if phase == 'train':
            if np.random.random() > 0.5:
                rcs_ifft = rcs_ifft[:, :, ::-1]    # 如果是训练集，则作数据增强，将512维度进行倒序翻转
                rcs_ifft = np.ascontiguousarray(rcs_ifft)  # 经上一步操作之后，numpy的地址不连续，转为tensor会报错，将numpy转为连续

        # set_trace()
        del ifft_Ev
        del ifft_Eh
        # 合并成2x401x512
        return rcs_ifft, ground_truth
