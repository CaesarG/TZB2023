import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from IPython.core.debugger import set_trace


class EvalModule(object):
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model

    def load_model(self, model, resume):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['model_state_dict']
        model.load_state_dict(state_dict_, strict=False)
        return model

    def eval_network(self, args):
        weight_path = r'weight_of_model'
        self.model = self.load_model(self.model, os.path.join(weight_path, args.resume))    # 载入训练模型权值
        self.model = self.model.to(self.device)    # 将模型载入GPU
        self.model.eval()

        dataset_module = self.dataset
        eval_dir = os.path.join(args.data_dir, args.test_txt)
        dsets_eval = dataset_module(annotation_lines=eval_dir, phase=args.phase)
        dsets_loader = DataLoader(dsets_eval,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
        len_dsets = len(dsets_loader)
        correct_num = 0
        pr_conf_sum = 0
        false_conf_sum = 0
        softmax = nn.Softmax(dim=1)
        Confusion_Matrix = np.zeros([10, 10])    # 混淆矩阵，第一维对应真实值，第二维对应预测值
        confusion_cnt = np.zeros(10)    # 对每个类别进行计数
        with torch.no_grad():
            for cnt, data in enumerate(dsets_loader):
                data_rcs = data[0]
                gt = data[1]
                data_rcs = data_rcs.to(device=self.device, non_blocking=True)
                pr_decs = self.model(data_rcs)
                pr_decs = softmax(pr_decs)
                pr_decs = pr_decs.squeeze()
                pr_sort_index = sorted(range(len(pr_decs)), key=lambda k: pr_decs[k], reverse=True)    # 降序排列预测结果
                pr_result = pr_sort_index[0]
                pr_conf = pr_decs[pr_result]

                confusion_cnt[gt] += 1
                Confusion_Matrix[gt][pr_result] += 1
                if pr_result == gt:
                    correct_num += 1
                    pr_conf_sum += pr_conf
                else:
                    false_conf_sum += pr_conf
                # set_trace()
                print('evaluating schedule: {0:d}/{1:d}'.format(cnt+1, len_dsets))    # 打印eval进度

        accuracy = correct_num/len_dsets*100
        avg_pr_conf = pr_conf_sum/correct_num
        avg_false_conf = false_conf_sum/(len_dsets-correct_num)
        print('The accuracy of classification is:\n{0:f}'.format(accuracy))
        print('The average correct prediction confidence is:\n{0:f}'.format(avg_pr_conf))
        print('The average false prediction confidence is:\n{0:f}'.format(avg_false_conf))
        for i in range(len(confusion_cnt)):
            Confusion_Matrix[i] = Confusion_Matrix[i] / confusion_cnt[i]
        print(Confusion_Matrix)
        return 0
