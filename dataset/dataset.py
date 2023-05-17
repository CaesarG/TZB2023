from torch.utils.data import Dataset
import scipy.io as sciio
import numpy as np
from IPython.core.debugger import set_trace

"""
注：生成标签annotation_lines为txt格式，每一行对应一帧的数据，分别记录帧文件路径和该帧类别
self.annotation_lines为一个数组，每一个元素记录原标签txt一行的字符串。
"""
class myDataset(Dataset):
    def __init__(self, annotation_lines, phase):
        super(myDataset, self).__init__()
        with open(annotation_lines, 'r') as f:
            self.annotation_lines = f.readlines()
        self.length = len(self.annotation_lines)
        self.phase = phase

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        rcs_abs, gt = self.get_rcs_abs(self.annotation_lines[index], self.phase)
        return rcs_abs, gt

    def get_rcs_abs(self, annotation_line, phase):
        line = annotation_line.split()
        path_rcs = line[0]
        ground_truth = int(line[1])    # 类别标签值
        data_rcs = sciio.loadmat(path_rcs)    # 为一个字典
        data_Ev = np.abs(data_rcs['frame_Ev']).astype(np.float32)
        data_Eh = np.abs(data_rcs['frame_Eh']).astype(np.float32)
        rcs_abs = np.concatenate((np.expand_dims(data_Ev, axis=0), np.expand_dims(data_Eh, axis=0)), axis=0)

        if phase == 'train':
            if np.random.random() > 0.5:
                rcs_abs = rcs_abs[:, :, ::-1]
        # set_trace()
        # 合并成2x401x512
        return rcs_abs, ground_truth
