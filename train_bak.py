import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loss import bce_loss
from IPython.core.debugger import set_trace


class TrainModule(object):
    def __init__(self, dataset, model):
        torch.manual_seed(317)
        self.dataset = dataset
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model

    # 训练模型保存，包含epoch，模型权值，优化器参数
    def save_model(self, path, epoch, model, optimizer):
        if isinstance(model, torch.nn.DataParallel):  # 多GPU时保存多加'module.'前缀
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            # 'loss': loss
        }, path)

    # 训练中断时，重载入模型权值文件，优化器参数，训练epoch
    def load_model(self, model, optimizer, resume, strict=True):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)  # 先将GPU权值文件转成CPU文件读入
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['model_state_dict']
        state_dict = {}
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = model.state_dict()
        if not strict:
            for k in state_dict:
                if k in model_state_dict:
                    if state_dict[k].shape != model_state_dict[k].shape:
                        print('Skip loading parameter {}, required shape{}, ' \
                              'loaded shape{}.'.format(k, model_state_dict[k].shape, state_dict[k].shape))
                        state_dict[k] = model_state_dict[k]
                else:
                    print('Drop parameter {}.'.format(k))
            for k in model_state_dict:
                if not (k in state_dict):
                    print('No param {}.'.format(k))
                    state_dict[k] = model_state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        return model, optimizer, epoch  # 此时的模型仍为CPU权值模型

    def train_network(self, args):

        self.optimizer = torch.optim.Adam(self.model.parameters(), args.init_lr)
        save_path = args.weight_save
        start_epoch = 1

        # 断点训练恢复
        if args.resume_train:
            self.model, self.optimizer, start_epoch = self.load_model(self.model,
                                                                      self.optimizer,
                                                                      args.resume_train,
                                                                      strict=True)
            start_epoch = start_epoch + 1
        # end

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.96, last_epoch=-1)

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if args.ngpus > 1:
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                self.model = nn.DataParallel(self.model)

        self.model.to(self.device)  # 将模型载入GPU

        criterion = bce_loss  # 使用交叉熵损失函数
        print('Setting up data...')

        # TODO 编写用于rcs的dataset，待完成
        dsets = {}
        dataset_module = self.dataset
        train_dir = args.data_dir + '/train.txt'
        val_dir = args.data_dir + '/val.txt'
        dsets['train'] = dataset_module(annotation_lines=train_dir)
        dsets['val'] = dataset_module(annotation_lines=val_dir)

        dsets_loader = {}
        dsets_loader['train'] = torch.utils.data.DataLoader(dsets['train'],
                                                            batch_size=args.batch_size,
                                                            shuffle=True,
                                                            num_workers=args.num_workers,
                                                            pin_memory=True,
                                                            drop_last=True)
        # collcate_fn=collacter  ,) 看到时是否需要根据rcs数据的特点自设collacter函数

        dsets_loader['val'] = torch.utils.data.DataLoader(dsets['val'],
                                                          batch_size=args.batch_size,
                                                          shuffle=True,
                                                          num_workers=args.num_workers,
                                                          pin_memory=True,
                                                          drop_last=True)
        # TODO dataset部分到此结束

        print('Starting training...')
        train_loss = np.array([])
        val_loss = np.array([])
        for epoch in range(start_epoch, args.num_epoch + 1):
            print('-' * 10)
            print('Epoch: {}/{} '.format(epoch, args.num_epoch))
            # 训练集训练
            epoch_loss = self.run_epoch(phase='train',
                                        data_loader=dsets_loader['train'],
                                        criterion=criterion)  # 进行一次epoch训练
            train_loss = np.append(train_loss, epoch_loss)  # 记录每次epoch损失值
            self.scheduler.step()  # 更新学习率 self.scheduler.step(epoch)

            # if epoch % 5 == 0 or epoch > 20:    # 存储权值文件
            # if epoch % 3 == 0:
            self.save_model(os.path.join(save_path, 'model_{}.pth'.format(epoch)),
                            epoch,
                            self.model,
                            self.optimizer)

            self.save_model(os.path.join(save_path, 'model_last.pth'),  # 存储最后一次epoch训练的权值文件
                            epoch,
                            self.model,
                            self.optimizer)
            # 验证集验证
            epoch_val_loss = self.run_epoch(phase='val',
                                            data_loader=dsets_loader['val'],
                                            criterion=criterion)  # 进行一次epoch训练
            val_loss = np.append(val_loss, epoch_val_loss)  # 记录每次epoch损失值

            # set_trace()

            train_val_loss = np.concatenate([np.expand_dims(train_loss, axis=1), np.expand_dims(val_loss, axis=1)],
                                            axis=1)

            # 每一次epoch都保存训练损失是防止训练中断时仍能保存最近一次epoch的损失
            np.savetxt(os.path.join(save_path, 'train_loss.txt'), train_val_loss, fmt='%.6f',
                       header='train_loss / val_loss')

    # TODO 根绝dataset和dataloader的取样方式，编写合适的训练代码，暂且按照：标签+一维距离像图的格式
    def run_epoch(self, phase, data_loader, criterion):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
        running_loss = 0.
        # TODO 修改data_loader结构
        for data, gt in data_loader:
            gt = gt.to(device=self.device, non_blocking=True)  # non_blocking一般与dataloader的pin_memory为True时为True
            # 配对使用。用以加速。
            data = data.to(device=self.device, non_blocking=True)
            if phase == 'train':
                self.optimizer.zero_grad()
                with torch.enable_grad():
                    pr_decs = self.model(data)
                    # set_trace()
                    loss = criterion(pr_decs, gt)  # 返回的loss为一个tensor
                    loss.backward()  # TODO 有报错——RuntimeError: CUDA error: device-side assert triggered
                    self.optimizer.step()
            else:
                with torch.no_grad():
                    pr_decs = self.model(data)
                    loss = criterion(pr_decs, gt)
            # TODO 修改至此结束

            running_loss += loss.item()
        epoch_loss = running_loss / len(data_loader)
        print('{} loss: {}'.format(phase, epoch_loss))
        return epoch_loss
