from os import listdir
from os.path import join, isfile, dirname, abspath
from glob import glob
from numpy import log10, abs, concatenate, expand_dims, float32
from numpy.fft import ifftshift, ifft
from torch import tensor, unsqueeze, no_grad
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

# ——————————————————————————————————————————————————————————————————————————————————————————————————————————
DIR = 'validDATA'
device = 'cpu'
if is_available():
    device = 'cuda:0'


# ——————————————————————————————————————————————————————————————————————————————————————————————————————————
class myDataset_ifft(Dataset):
    def __init__(self, path):
        super(myDataset_ifft, self).__init__()
        self.path = path
        self.length = len([f for f in listdir(path) if isfile(join(path, f))])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rcs_ifft = self.get_rcs_ifft(index)
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
        return rcs_ifft


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
            self.worker = Worker(dir)
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
            loc = dirname(abspath(__file__))
            self.label.setText('Excel文件生成成功，文件所在目录为 {}'.format(loc))
            self.btn.setEnabled(True)


class Worker(QThread):
    progress_signal = pyqtSignal(int)  # 定义一个信号，用于更新进度条

    success_signal = pyqtSignal(bool)

    def __init__(self, dir=DIR):
        super(QThread, self).__init__()
        self.dir = dir

    def pd_toExcel(self, data, fileName='卷就不队_电子科技大学_测试结果.xlsx'):
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
        self.success_signal.emit(True)

    def run(self):
        self.success_signal.emit(False)
        model = load('Tesla.dll')
        model.to(device)
        model.eval()
        dataset = myDataset_ifft
        dsets = dataset(path=self.dir)

        softmax = Softmax(dim=1)

        dsets_len = dsets.length
        ans = []
        with no_grad():
            for i in range(dsets_len):
                res = [i + 1, 'frame_{}'.format(i + 1)]
                data_rcs = dsets[i]
                # print(data_rcs)
                data_rcs = data_rcs.to(device=device, non_blocking=True)
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
                self.progress_signal.emit((i + 1) * 100 // dsets_len)  # 发送信号，更新进度条
        ans = list(map(list, zip(*ans)))
        self.pd_toExcel(ans)


if __name__ == '__main__':
    app = QApplication(argv)
    ex = Example()
    exit(app.exec_())
