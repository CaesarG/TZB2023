import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import scipy.io as sciio
import pandas as pd
from tqdm import trange

# ——————————————————————————————————————————————————————————————————————————————————————————————————————————
DIR = 'validDATA'
MODEL = 'Tesla.dll'


# ——————————————————————————————————————————————————————————————————————————————————————————————————————————


class myDataset_ifft(Dataset):
    def __init__(self, path):
        super(myDataset_ifft, self).__init__()
        self.path = path
        self.length = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rcs_ifft = self.get_rcs_ifft(index)
        rcs_ifft = torch.tensor(rcs_ifft)
        rcs_ifft = torch.unsqueeze(rcs_ifft, dim=0)
        return rcs_ifft

    def get_rcs_ifft(self, index):
        path_rcs = os.path.join(self.path, 'frame_{}.mat'.format(index + 1))
        data_rcs = sciio.loadmat(path_rcs)  # sciio加载的每帧mat数据为一个字典
        data_Ev = data_rcs['frame_Ev'].astype(complex)
        data_Eh = data_rcs['frame_Eh'].astype(complex)

        ifft_Ev = np.log10(np.abs(np.fft.ifftshift(np.fft.ifft(data_Ev.T), axes=1)).astype(np.float32))
        ifft_Eh = np.log10(np.abs(np.fft.ifftshift(np.fft.ifft(data_Eh.T), axes=1)).astype(np.float32))

        rcs_ifft = np.concatenate((np.expand_dims(ifft_Ev.T, axis=0), np.expand_dims(ifft_Eh.T, axis=0)), axis=0)

        # set_trace()
        del ifft_Ev
        del ifft_Eh
        # 合并成2x401x512
        return rcs_ifft


def pd_toExcel(data, fileName='卷就不队_电子科技大学_测试结果.xlsx'):
    dfData = {
        '序号': data[0],
        '测试数据块名称': data[1],
        '识别结果': data[2],
        '单个数据块识别概率': data[3]
    }
    df = pd.DataFrame(dfData)
    writer = pd.ExcelWriter(fileName, engine='xlsxwriter')
    df.style.set_properties(**{'text-align': 'center', 'border': '1px solid black'}).to_excel(writer,
                                                                                              sheet_name='Sheet1',
                                                                                              index=False)
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    worksheet.set_column('A:C', 15)
    worksheet.set_column('D:D', 20)
    format_border = workbook.add_format({'border': 1})
    worksheet.conditional_format('A1:XFD1048576', {'type': 'no_blanks', 'format': format_border})
    # 保存 Excel 文件
    writer.close()


if __name__ == '__main__':

    dir = input('请输入ValidDATA文件夹的绝对路径，以回车结束（例如: E:/validDATA):')
    if dir.strip() == '':
        dir = DIR
    model = torch.jit.load(MODEL)
    model.to('cuda:0')
    model.eval()
    dataset = myDataset_ifft
    dsets = dataset(path=dir)

    softmax = nn.Softmax(dim=1)

    ans = []
    with torch.no_grad():
        for i in trange(dsets.length):
            res = [i + 1, 'frame_{}'.format(i + 1)]
            data_rcs = dsets[i]
            # print(data_rcs)
            data_rcs = data_rcs.to(device='cuda:0', non_blocking=True)
            pr_decs = model(data_rcs)
            pr_decs = softmax(pr_decs)
            pr_decs = pr_decs.squeeze()
            pr_sort_index = sorted(range(len(pr_decs)), key=lambda k: pr_decs[k], reverse=True)  # 降序排列预测结果
            pr_result = pr_sort_index[0]
            pr_conf = pr_decs[pr_result]
            # TODO: 拒判

            res.append(pr_result)
            res.append(pr_conf.item())
            ans.append(res)
    ans = list(map(list, zip(*ans)))
    pd_toExcel(ans)
