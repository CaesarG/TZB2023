from torch.utils.data import Dataset
import scipy.io as sciio
import numpy as np
from IPython.core.debugger import set_trace

"""
注：生成标签annotation_lines为txt格式，每一行对应一帧的数据，分别记录帧文件路径和该帧类别
self.annotation_lines为一个数组，每一个元素记录原标签txt一行的字符串。
"""
class myDataset_4(Dataset):
    def __init__(self, annotation_lines, phase):
        super(myDataset_4, self).__init__()
        with open(annotation_lines, 'r') as f:
            self.annotation_lines = f.readlines()
        self.length = len(self.annotation_lines)
        self.phase = phase

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        rcs_4_channel, gt = self.get_rcs_abs(self.annotation_lines[index], self.phase)
        return rcs_4_channel, gt

    def get_rcs_abs(self, annotation_line, phase):
        line = annotation_line.split()
        path_rcs = line[0]
        ground_truth = int(line[1])    # 类别标签值
        data_rcs = sciio.loadmat(path_rcs)    # 为一个字典
        data_Ev = data_rcs['frame_Ev'].astype(np.complex)
        data_Eh = data_rcs['frame_Eh'].astype(np.complex)
        data_Ev_real = np.real(data_Ev).astype(np.float32)
        data_Ev_imag = np.imag(data_Ev).astype(np.float32)
        data_Eh_real = np.real(data_Eh).astype(np.float32)
        data_Eh_imag = np.imag(data_Eh).astype(np.float32)
        rcs_2_channel = np.concatenate((np.expand_dims(data_Ev_real, axis=0), np.expand_dims(data_Ev_imag, axis=0)), axis=0)
        rcs_3_channel = np.concatenate((rcs_2_channel, np.expand_dims(data_Eh_real, axis=0)), axis=0)
        rcs_4_channel = np.concatenate((rcs_3_channel, np.expand_dims(data_Eh_imag, axis=0)), axis=0)

        if phase == 'train':
            if np.random.random() > 0.5:
                rcs_4_channel = rcs_4_channel[:, :, ::-1]
        # set_trace()
        # 合并成4x401x512
        return rcs_4_channel, ground_truth
