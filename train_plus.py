import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loss import bce_loss
from IPython.core.debugger import set_trace
from sam import SAM
from utility.bypass_bn import enable_running_stats,disable_running_stats
from utility.smooth_cross_entropy import smooth_crossentropy

class TrainModule(object):
    def __init__(self, dataset, model):
        torch.manual_seed(317)
        self.dataset = dataset
        self.num_classes = 10
        self.lamda = 0.5 #第一个调的地方0.1,0.5,1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.anchors = nn.Parameter(torch.zeros(self.num_classes,self.num_classes).double())

    def set_anchors(self,means):
        self.anchors = nn.Parameter(means.double())


    def distance_classifier(self,x):
        n = x.size(0)
        m = self.num_classes
        d = self.num_classes

        x = x.unsqueeze(1).expand(n,m,d).double()
        anchors = self.anchors.unsqueeze(0).expand(n,m,d).cuda()
        dists = torch.norm(x-anchors,2,2)

        return dists
    
    def CACloss(self,distance,gt):
        true = torch.gather(distance,1,gt.view(-1,1)).view(-1)
        non_gt = torch.Tensor([[i for i in range(self.num_classes) if gt[x] != i] for x in range(len(distance))]).long().cuda()
        others = torch.gather(distance,1,non_gt)

        anchor = torch.mean(true)
        tuplet = torch.exp(-others+true.unsqueeze(1))
        tuplet = torch.mean(torch.log(1+torch.sum(tuplet,dim = 1)))
        total = self.lamda*anchor + tuplet

        return total,anchor,tuplet
    # 训练模型保存，包含epoch，模型权值，优化器参数
    def save_model(self, path, epoch, model, optimizer):
        if isinstance(model, torch.nn.DataParallel):    # 多GPU时保存多加'module.'前缀
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
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)    # 先将GPU权值文件转成CPU文件读入
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
        return model, optimizer, epoch    # 此时的模型仍为CPU权值模型

    def train_network(self, args):
        alpha = 10 #第二个调的地方5,10
        anchors = torch.diag(torch.Tensor([alpha for i in range(self.num_classes)]))
        self.set_anchors(anchors) 
        base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
        self.optimizer = SAM(self.model.parameters(), base_optimizer, lr=args.init_lr, momentum=0.9)
        
        # self.optimizer = torch.optim.Adam(self.model.parameters(), args.init_lr)
        save_path = args.weight_save
        start_epoch = 1

        # 断点训练恢复
        if args.resume_train:
            self.model, self.optimizer, start_epoch = self.load_model(self.model,
                                                                      self.optimizer,
                                                                      args.resume_train,
                                                                      strict=True)
            start_epoch = start_epoch+1
        # end

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.96, last_epoch=-1)

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if args.ngpus > 1:
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                self.model = nn.DataParallel(self.model)
                
        self.model.to(self.device)    # 将模型载入GPU

        # criterion = bce_loss  # 使用交叉熵损失函数
        criterion = self.CACloss
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
                                                            #collcate_fn=collacter  ,) 看到时是否需要根据rcs数据的特点自设collacter函数

        dsets_loader['val'] = torch.utils.data.DataLoader(dsets['val'],
                                                            batch_size=args.batch_size,
                                                            shuffle=True,
                                                            num_workers=args.num_workers,
                                                            pin_memory=True,
                                                            drop_last=True)
        # TODO dataset部分到此结束

        print('Starting training...')
        best_acc = 0
        for epoch in range(start_epoch, args.num_epoch + 1):
            print('-' * 10)
            print('Epoch: {}/{} '.format(epoch, args.num_epoch))
            # 训练集训练
            epoch_train_acc = self.run_epoch(phase='train',
                                        data_loader=dsets_loader['train'],
                                        criterion=criterion)    # 进行一次epoch训练


            # if epoch % 5 == 0 or epoch > 20:    # 存储权值文件
            # if epoch % 3 == 0:
            # self.save_model(os.path.join(save_path, 'model_{}.pth'.format(epoch)),
            #                     epoch,
            #                     self.model,
            #                     self.optimizer)

            # self.save_model(os.path.join(save_path, 'model_last.pth'),    # 存储最后一次epoch训练的权值文件
            #                 epoch,
            #                 self.model,
            #                 self.optimizer)
            # 验证集验证
            epoch_val_acc = self.run_epoch(phase='val',
                                            data_loader=dsets_loader['val'],
                                            criterion=criterion)  # 进行一次epoch训练

            self.scheduler.step()    # 更新学习率 self.scheduler.step(epoch)
            if best_acc < epoch_val_acc:
                best_acc = epoch_val_acc
                self.save_model(os.path.join(save_path, 'model_best_{}.pth'.format(epoch)),
                                epoch,
                                self.model,
                                self.optimizer)

    # TODO 根绝dataset和dataloader的取样方式，编写合适的训练代码，暂且按照：标签+一维距离像图的格式
    def run_epoch(self, phase, data_loader, criterion):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
        running_loss = 0
        train_loss = 0
        correctDist = 0
        total = 0
        batch_idx = 0
        Test_Loss = 0
        scores = []
        predicts = []
        y_gt = []
        acc = 0
        # TODO 修改data_loader结构
        if phase == 'train':
            for batch_idx,(data, gt) in enumerate(data_loader):
                gt = gt.to(device=self.device, non_blocking=True)    # non_blocking一般与dataloader的pin_memory为True时为True
                                                                    # 配对使用。用以加速。
                data = data.to(device=self.device, non_blocking=True)
                
                self.optimizer.zero_grad()
                with torch.enable_grad():
                    enable_running_stats(self.model)
                    pr_decs = self.model(data)
                    distance = self.distance_classifier(pr_decs)
                    loss,anchorloss,tupletloss = criterion(distance,gt)
                    # set_trace()
                    #loss = criterion(pr_decs, gt)    # 返回的loss为一个tensor
                    loss.backward()    # TODO 有报错——RuntimeError: CUDA error: device-side assert triggered
                    # self.optimizer.step()
                    self.optimizer.first_step(zero_grad=True) 
                    disable_running_stats(self.model)
                    # criterion(self.model(data), gt).backward() 
                    sec_pr_decs = self.model(data)
                    sec_dis = self.distance_classifier(sec_pr_decs)
                    sec_loss,anchorloss,tupletloss=criterion(sec_dis,gt)
                    sec_loss.backward()
                    self.optimizer.second_step(zero_grad=True)                  
                    train_loss += loss.item()
                    _, predicted = distance.min(1)
                    total += gt.size(0)
                    correctDist += predicted.eq(gt).sum().item()
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
        % (train_loss/(batch_idx+1), 100.*correctDist/total, correctDist, total))
            epoch_acc = 100.*correctDist/total
        else:
            with torch.no_grad():
                for batch_idx,(data, gt) in enumerate(data_loader):
                    gt = gt.to(device=self.device, non_blocking=True)    # non_blocking一般与dataloader的pin_memory为True时为True
                                                                        # 配对使用。用以加速。
                    data = data.to(device=self.device, non_blocking=True)
                    pr_decs = self.model(data)
                    distance = self.distance_classifier(pr_decs)
                    loss,anchorloss,tupletloss = self.CACloss(distance,gt)
                    
                    Test_Loss += loss.item()
                    _, predicted = distance.min(1)
                    
                    total += gt.size(0)

                    acc += predicted.eq(gt).cpu().sum()
                    scores.append(distance)
                    predicts.append(predicted)
                    y_gt.append(gt)
            print('Test_Loss: %.3f | Acc: %.3f%% (%d/%d)'
        % (Test_Loss/(batch_idx+1), 100.*acc/total, acc, total))
            epoch_acc = 100. * acc / total
        # TODO 修改至此结束


        return epoch_acc
