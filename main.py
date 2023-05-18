import argparse
import torch
import train
import test
import eval
from dataset.dataset import myDataset
from dataset.dataset_4 import myDataset_4
from dataset.dataset_ifft import myDataset_ifft
from dataset.dataset_ifft_4 import myDataset_ifft_4
from model.DPA_Alexnet import DPAAlexNet
from model.DPA_Alexnet_4channel import DPAAlexNet_4channel
from model.vgg import vgg

# 调试函数
from IPython.core.debugger import set_trace


def parse_args():
    parser = argparse.ArgumentParser(description='Alexnet Implementation')
    parser.add_argument('--channel', type=int, default=4, help='Number of channel')
    parser.add_argument('--ifft', type=str, default='False', help='Whether do ifft transform')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--num_epoch', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='Num_workers for dataloader')
    parser.add_argument('--model', type=str, default='vgg', help='model of backbone: vgg or Alexnet')
    parser.add_argument('--init_lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--ngpus', type=int, default=1, help='Number of gpus, ngpus>1 for multigpu')
    parser.add_argument('--resume_train', type=str, default='', help='Weights resumed in training/Absolute path')
    parser.add_argument('--resume', type=str, default='model_10.pth', help='Weights resumed testing and evaluation')
    parser.add_argument('--weight_save', type=str, default='weight_of_model', help='Weights saved directory')
    parser.add_argument('--data_dir', type=str, default='dataRCS/annotations', help='Path of data and annotation '
                                                                                    'directory')
    parser.add_argument('--test_txt', type=str, default='test.txt', help='The name of test file in dataRCS/annotations '
                                                                         'for test or eval')
    parser.add_argument('--phase', type=str, default='test', help='Phase choice= {train, test, eval}')
    parser.add_argument('--CA', action="store_true", default='False', help='Whether to use CA Attention')
    # parser.add_argument('--conf_thresh', type=float, default=0.3, help='Confidence thresh hold')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    num_classes = args.num_classes
    if args.channel == 2:
        # 模型选择
        if args.model == 'Alexnet':
            model = DPAAlexNet(num_classes=num_classes)
        else:
            # VGG模型还未加入DPA模块
            # 需要使用使用vgg的何种模型就将moedl_name后面修改成何种(如vgg16)，同时torch.load的权值文件也要替换为相应模型
            model = vgg(model_name='vgg19', num_classes=num_classes, init_weights=True, CA=args.CA)
            pretrained_dict = torch.load('model/pretrain_weight/vgg19-dcbb9e9d.pth')  # 加载预训练权重模型(.pth文件)参数
            model_dict = model.state_dict()  # 得到模型的参数字典

            # 判断预训练模型中网络的模块是否修改后的网络中也存在，并且shape相同，如果相同则取出
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                                   k in model_dict and (v.shape == model_dict[k].shape)}

            # 更新修改之后的 model_dict
            model_dict.update(pretrained_dict)

            # 加载我们真正需要的 state_dict
            model.load_state_dict(model_dict, strict=False)

        # 是否ifft
        if args.ifft == 'False':
            dataset = myDataset
        else:
            dataset = myDataset_ifft
    elif args.channel == 4:
        if args.ifft == 'False':
            dataset = myDataset_4
        else:
            dataset = myDataset_ifft_4
        if args.model == 'Alexnet':
            model = DPAAlexNet_4channel(num_classes=num_classes)
        else:
            print('no 4 channel model for vgg temporarily')
    else:
        print('invalid num of channel')

    if args.phase == 'train':
        rcs = train.TrainModule(dataset=dataset, model=model)
        rcs.train_network(args)
    elif args.phase == 'test':
        rcs = test.TestModule(dataset=dataset, model=model)
        rcs.test_network(args)
    elif args.phase == 'eval':
        rcs = eval.EvalModule(dataset=dataset, model=model)
        rcs.eval_network(args)
