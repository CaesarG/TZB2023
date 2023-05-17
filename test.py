import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from IPython.core.debugger import set_trace


class TestModule(object):
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

    def test_network(self, args):
        weight_path = r'D:\TZB\Alexnet\weight_of_model'
        self.model = self.load_model(self.model, os.path.join(weight_path, args.resume))    # 载入训练模型权值
        self.model = self.model.to(self.device)    # 将模型载入GPU
        self.model.eval()


        dataset_module = self.dataset
        test_dir = os.path.join(args.data_dir, args.test_txt)
        dsets_test = dataset_module(annotation_lines=test_dir, phase=args.phase)
        dsets_loader = DataLoader(dsets_test,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
        save_dir = r'D:\TZB\Alexnet\test_results'
        pr_results = np.array([])
        pr_confs = np.array([])
        gts = np.array([])
        len_dsets = len(dsets_loader)
        softmax = nn.Softmax(dim=1)
        for cnt, data in enumerate(dsets_loader):
            data_rcs = data[0]
            gt = data[1]
            data_rcs = data_rcs.to(device=self.device, non_blocking=True)
            pr_decs = self.model(data_rcs)
            pr_decs = softmax(pr_decs)
            pr_decs = pr_decs.squeeze()
            pr_sort_index = sorted(range(len(pr_decs)), key=lambda k: pr_decs[k], reverse=True)
            pr_result = pr_sort_index[0]
            pr_conf = pr_decs[pr_result]
            # set_trace()    # debug时使用
            pr_results = np.append(pr_results, pr_result)
            pr_confs = np.append(pr_confs, pr_conf)
            gts = np.append(gts, gt)
            print('testing schedule: {0:d}/{1:d}'.format(cnt, len_dsets))    # 打印test进度
        pr_conf = np.concatenate([np.expand_dims(pr_results, axis=1), np.expand_dims(pr_confs, axis=1)], axis=1)
        pr_conf_gt = np.concatenate([pr_conf, np.expand_dims(gts, axis=1)], axis=1)

        np.savetxt(os.path.join(save_dir, args.test_txt), pr_conf_gt, fmt='%d',
                   header='prediction /confidence / ground truth')
        return 0






