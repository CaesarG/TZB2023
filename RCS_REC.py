# from os import listdir
# import time
from os.path import join, dirname, abspath, exists
from glob import glob
from numpy import log10, abs, concatenate, expand_dims, float32, min, max, argmin
from numpy.fft import ifftshift, ifft
from torch import tensor, no_grad, float64, unsqueeze
import torch.jit
from torch.jit import load
from torch.nn import Softmax
from torch.cuda import is_available
from torch.utils.data import Dataset
from scipy.io import loadmat
from pandas import DataFrame, ExcelWriter
from sys import argv, exit
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, \
    QDesktopWidget, QProgressBar
from PyQt5.QtCore import QThread, pyqtSignal

# from multiprocessing import cpu_count

# ——————————————————————————————————————————————————————————————————————————————————————————————————————————
DIR = 'validDATA'
device = 'cpu'
if is_available():
    device = 'cuda:0'

anchors = tensor([[1.0074e+01, -7.0760e-02, 3.4007e-02, -6.0953e-02, -2.8561e-02,
                   -4.7631e-02, -1.1976e-01, -4.5914e-02, 7.9931e-02, 4.5816e-03],
                  [-1.0493e-01, 1.0158e+01, 1.6512e-03, 4.0228e-02, -9.6871e-03,
                   -1.5703e-02, -5.0202e-02, -3.1814e-02, -6.8128e-02, -4.7713e-02],
                  [-3.3110e-02, -7.3210e-02, 9.9843e+00, 5.2903e-02, -2.6644e-02,
                   -5.1030e-02, 2.9400e-02, -5.6540e-02, 5.8466e-02, 2.8036e-02],
                  [6.2251e-02, -1.4779e-03, 1.1965e-02, 1.0073e+01, -2.2256e-02,
                   -3.8470e-02, -6.8585e-02, 6.2917e-02, 1.1223e-01, 5.6566e-03],
                  [-4.8107e-02, -3.2796e-02, -6.5273e-02, 5.9005e-03, 1.0275e+01,
                   1.9821e-02, -1.0396e-01, -4.6109e-02, 7.6573e-03, 7.5969e-03],
                  [2.4843e-02, -3.2213e-02, -1.1588e-02, 2.7143e-02, 4.1448e-02,
                   9.9014e+00, -9.9555e-03, 3.9144e-02, 6.4675e-02, 3.2475e-02],
                  [9.2964e-02, 5.0821e-04, -1.3793e-01, -1.2574e-02, 1.0644e-02,
                   -4.0691e-02, 1.0028e+01, -8.5908e-02, 7.4214e-02, -4.0527e-02],
                  [1.5421e-03, 1.0801e-01, -1.2993e-01, -1.4387e-02, -1.7921e-01,
                   -4.5816e-02, -2.1667e-02, 1.0621e+01, 5.8735e-02, -1.1873e-01],
                  [2.6066e-02, 6.4731e-02, -3.5778e-02, 1.7572e-03, 8.2858e-02,
                   -5.5604e-02, 8.3159e-03, -8.5511e-02, 1.0324e+01, 4.4276e-02],
                  [-1.2988e-02, -1.0416e-01, 1.2649e-01, 6.6912e-02, 1.0347e-01,
                   1.1866e-02, -5.2605e-02, -1.8179e-02, -2.1181e-02, 9.7523e+00]],
                 device='cuda:0', dtype=float64)


# ——————————————————————————————————————————————————————————————————————————————————————————————————————————
class myDataset_ifft(Dataset):
    def __init__(self, path, length):
        super(myDataset_ifft, self).__init__()
        self.path = path
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rcs_ifft, gt = self.get_rcs_ifft(index)
        rcs_ifft = tensor(rcs_ifft)
        rcs_ifft = unsqueeze(rcs_ifft, dim=0)
        return rcs_ifft

    def get_rcs_ifft(self, index):
        path_rcs = join(self.path, 'frame_{}.mat'.format(index + 1))
        data_rcs = loadmat(path_rcs)  # sciio加载的每帧mat数据为一个字典
        data_Ev = data_rcs['frame_Ev'].astype(complex)
        data_Eh = data_rcs['frame_Eh'].astype(complex)

        ifft_Ev = log10(abs(ifftshift(ifft(data_Ev.T), axes=1)).astype(float32))
        ifft_Eh = log10(abs(ifftshift(ifft(data_Eh.T), axes=1)).astype(float32))

        rcs_ifft = concatenate((expand_dims(ifft_Ev.T, axis=0), expand_dims(ifft_Eh.T, axis=0)), axis=0)

        # set_trace()
        del ifft_Ev
        del ifft_Eh
        # 合并成2x401x512
        return rcs_ifft, 1


def judge(dir):
    suffix = '.mat'
    filtered_files = glob(join(dir, '*' + suffix))
    return filtered_files


class Example(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        # 创建标签、输入框和按钮
        self.lbl = QLabel('请输入ValidDATA文件夹的绝对路径(例如: E:\\validDATA):', self)
        self.edit = QLineEdit(self)
        self.btn = QPushButton('启动目标识别', self)
        self.label = QLabel(self)
        self.dir_label = QLabel(self)

        self.progresslabel = QLabel('正在进行目标识别...', self)
        self.progressbar = QProgressBar()
        self.progress = QHBoxLayout()
        self.progress.addWidget(self.progresslabel)
        self.progress.addWidget(self.progressbar)

        # 创建垂直布局管理器
        vbox = QVBoxLayout()
        vbox.addWidget(self.lbl)
        vbox.addWidget(self.edit)
        vbox.addWidget(self.dir_label)
        vbox.addWidget(self.btn)
        vbox.addLayout(self.progress)
        vbox.addWidget(self.label)

        # 创建水平布局管理器
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addLayout(vbox)
        hbox.addStretch(1)

        # 设置主窗口的布局管理器
        self.setLayout(hbox)

        # 设置窗口标题和大小
        self.setWindowTitle('基于雷达RCS数据的空间物体智能识别器')
        self.resize(500, 150)

        # main
        self.progresslabel.setVisible(False)
        self.progressbar.setVisible(False)
        self.label.setVisible(False)
        self.dir_label.setVisible(False)
        self.btn.clicked.connect(self.on_button_click)

        self.center()
        # 显示窗口
        self.show()

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) // 2,
                  (screen.height() - size.height()) // 2)

    def on_button_click(self):
        self.label.setVisible(False)
        dir = self.edit.text()
        dir_path = dir
        if dir.strip() == '':
            dir = DIR
            loc = dirname(abspath(__file__))
            dir_path = join(loc, DIR)
        # print(dir)
        self.dir_label.setVisible(True)
        if judge(dir):
            self.dir_label.setText('您的测试文件所在目录为 {}'.format(dir_path))
            # 创建一个 Worker 对象，并连接其信号到槽函数
            self.worker = Worker(len(judge(dir)), dir)
            self.worker.progress_signal.connect(self.update_progress)
            self.worker.success_signal.connect(self.update_success)
            self.progresslabel.setVisible(True)
            self.progressbar.setVisible(True)
            self.progressbar.setValue(0)
            self.btn.setEnabled(False)
            # 启动线程
            self.worker.start()
        else:
            self.dir_label.setText('"{}"该文件夹中无*.mat文件，请重新输入后点击"启动识别"按钮'.format(dir_path))

    def update_progress(self, value):
        # 更新进度条的值
        self.progressbar.setValue(value)

    def update_success(self, value):
        if value:
            self.progresslabel.setVisible(False)
            self.progressbar.setVisible(False)
            self.label.setVisible(True)
            if value == 1:
                loc = dirname(abspath(__file__))
                self.label.setText('Excel文件生成成功，文件所在目录为 {}'.format(loc))
            else:
                self.label.setText('所要求的Excel已被打开，请关闭当前Excel文件后重新进行目标识别')
            self.btn.setEnabled(True)


def distance_classifier(x):
    n = 1
    m = 10
    d = 10

    x = x.unsqueeze(1).expand(n, m, d).double()
    A = anchors.unsqueeze(0).expand(n, m, d).cuda()

    dists = torch.norm(x - A, 2, 2)
    return dists


class Worker(QThread):
    progress_signal = pyqtSignal(int)  # 定义一个信号，用于更新进度条

    success_signal = pyqtSignal(int)

    def __init__(self, length, dir=DIR):
        super(QThread, self).__init__()
        self.dir = dir
        dataset = myDataset_ifft
        self.dsets = dataset(path=self.dir, length=length)
        self.threshold = 1.25

    def pd_toExcel(self, data, fileName='卷就不队_电子科技大学_测试结果.xlsx'):
        if os.path.exists('~$' + fileName):
            self.success_signal.emit(2)
            return
        dfData = {
            '序号': data[0],
            '测试数据块名称': data[1],
            '识别结果': data[2],
            '单个数据块识别概率': data[3]
        }
        df = DataFrame(dfData)
        writer = ExcelWriter(fileName, engine='xlsxwriter')
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
        self.success_signal.emit(1)

    def get_predicted(self, total_scores):

        min_score = min(total_scores)
        predicted = argmin(total_scores)
        if min_score > self.threshold:
            predicted = 10
        return predicted

    def run(self):
        model = load('Tesla.dll')
        model.to(device)
        model.eval()

        # NUM_WORKERS = cpu_count() - 2
        self.success_signal.emit(False)

        softmax = Softmax(dim=1)

        dsets_len = self.dsets.length
        # dataloader = torch.utils.data.DataLoader(
        #     self.dsets,
        #     batch_size=1,
        #     shuffle=False,
        #     num_workers=NUM_WORKERS,
        #     pin_memory=True,
        #     drop_last=True,
        # )
        ans = []
        with no_grad():
            for i in range(dsets_len):
                data_rcs = self.dsets[i]
                res = [i + 1, 'frame_{}'.format(i + 1)]
                data_rcs = data_rcs.to(device=device, non_blocking=True)
                pr_decs = model(data_rcs)
                # pr_decs = softmax(pr_decs)
                # pr_decs = pr_decs.squeeze()
                # pr_sort_index = sorted(range(len(pr_decs)), key=lambda k: pr_decs[k], reverse=True)  # 降序排列预测结果
                # pr_result = pr_sort_index[0]
                # pr_conf = pr_decs[pr_result]
                distances = distance_classifier(pr_decs)
                softmin = softmax(-distances)
                invScores = 1 - softmin
                scores = distances * invScores
                # print(i, softmax(pr_decs), scores)
                pr_result = self.get_predicted(scores.cpu().detach().numpy().flatten())

                # TODO: 拒判
                if pr_result < 10:
                    pr_result += 1
                else:
                    pr_result = 0
                res.append(pr_result)
                res.append(max(softmin.cpu().detach().numpy().flatten()))
                ans.append(res)
                self.progress_signal.emit((i + 1) * 100 // dsets_len)  # 发送信号，更新进度条
        ans = list(map(list, zip(*ans)))
        self.pd_toExcel(ans)


if __name__ == '__main__':
    app = QApplication(argv)
    ex = Example()
    exit(app.exec_())
