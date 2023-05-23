import argparse
import torch
import torchvision
import train
import test
import eval
from dataset.dataset import myDataset
from dataset.dataset_4 import myDataset_4
from dataset.dataset_ifft import myDataset_ifft
from model.DPA_Alexnet import DPAAlexNet
from model.DPA_Alexnet_4channel import DPAAlexNet_4channel
from model.vgg import vgg
from Autoformer.supernet_transformer import Vision_TransformerSuper
from lib.config import cfg, update_config_from_file
# 调试函数
from IPython.core.debugger import set_trace
import random
import timm
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed

from model.effnetv2 import effnetv2_s
# from timm.models.resnet import *
# from

def sample_configs(choices):

    config = {}
    dimensions = ['mlp_ratio', 'num_heads']
    depth = random.choice(choices['depth'])
    for dimension in dimensions:
        config[dimension] = [random.choice(choices[dimension]) for _ in range(depth)]

    config['embed_dim'] = [random.choice(choices['embed_dim'])]*depth

    config['layer_num'] = depth
    return config

def parse_args():
    parser = argparse.ArgumentParser(description='Alexnet Implementation')
    parser.add_argument('--channel', type=int, default=4, help='Number of channel')
    parser.add_argument('--ifft', type=str, default='False', help='Whether do ifft transform')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--num_epoch', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of batch size')
    parser.add_argument('--num_workers', type=int, default=23, help='Num_workers for dataloader')
    parser.add_argument('--model', type=str, default='vgg', help='model of backbone: vgg or Alexnet')
    parser.add_argument('--init_lr', type=float, default=3e-5, help='Initial learning rate')
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
    # parser.add_argument('--cfg',default="./experiments/supernet/supernet-T.yaml",help='experiment configure file name',required=True,type=str)
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)') 
    parser.add_argument('--max_relative_position', type=int, default=14, help='max distance in relative position embedding')
    parser.add_argument('--relative_position', action='store_true')
    parser.add_argument('--no_abs_pos', default='True', action='store_true')
    parser.add_argument('--gp', action='store_true')
    parser.add_argument('--change_qkv',default='True',action='store_true')
    # parser.add_argument('--conf_thresh', type=float, default=0.3, help='Confidence thresh hold')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    num_classes = args.num_classes
    if args.channel == 2:
        # 模型选择
        # print('Avaliable models on timm')
        # print(timm.list_models())
        if args.model == 'Alexnet':
            model = DPAAlexNet(num_classes=num_classes)
        elif args.model == 'vgg':
            # VGG模型还未加入DPA模块
            # 需要使用使用vgg的何种模型就将moedl_name后面修改成何种(如vgg16)，同时torch.load的权值文件也要替换为相应模型
            model = vgg(model_name='vgg11', num_classes=num_classes, init_weights=True, CA=args.CA)
            pretrained_dict_path=r'model/pretrain_weight/vgg11-bbd30ac9.pth'
            # pretrained_dict = torch.load(pretrained_dict_path)  # 加载预训练权重模型(.pth文件)参数
            model_dict = model.state_dict()  # 得到模型的参数字典

            # 判断预训练模型中网络的模块是否修改后的网络中也存在，并且shape相同，如果相同则取出
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                                   k in model_dict and (v.shape == model_dict[k].shape)}

            # 更新修改之后的 model_dict
            model_dict.update(pretrained_dict)

            # 加载我们真正需要的 state_dict
            model.load_state_dict(model_dict, strict=False)
        elif args.model == 'autoformer':
            # todo: need to add model config for each model however the config is not necessary
            # unless vit is used
            update_config_from_file(args.cfg)
            choices = {'num_heads': cfg.SEARCH_SPACE.NUM_HEADS, 'mlp_ratio': cfg.SEARCH_SPACE.MLP_RATIO,
                    'embed_dim': cfg.SEARCH_SPACE.EMBED_DIM , 'depth': cfg.SEARCH_SPACE.DEPTH}
            config=sample_configs(choices=choices)
            model = Vision_TransformerSuper(img_size=(401,512),
                                            patch_size=(20,32),
                                            embed_dim=cfg.SUPERNET.EMBED_DIM, depth=cfg.SUPERNET.DEPTH,
                                            num_heads=cfg.SUPERNET.NUM_HEADS,mlp_ratio=cfg.SUPERNET.MLP_RATIO,
                                            qkv_bias=True, drop_rate=args.drop,
                                            drop_path_rate=args.drop_path,
                                            gp=args.gp,
                                            num_classes=args.num_classes,
                                            max_relative_position=args.max_relative_position,
                                            relative_position=args.relative_position,
                                            change_qkv=args.change_qkv, abs_pos=not args.no_abs_pos)
            model.set_sample_config(config=config)
        elif args.model == 'efficientnet':
            model = timm.create_model('efficientnet_b1', pretrained=True,num_classes=10)
            model.conv_stem=nn.Conv2d(2,32,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=False)
        elif args.model == 'resnet':
            model = timm.create_model('resnet50',pretrained=False,num_classes=10)
            model.conv1=nn.Conv2d(2,64,kernel_size=(3,3),stride=(1,1),padding=(3,3),bias=False)
        elif args.model == 'vit':
            model = timm.create_model('vit_small_patch16_224',pretrained=True,num_classes=10)
            model.patch_embed.proj=nn.Conv2d(2,768,kernel_size=(16,16),stride=(16,16))
            model.patch_embed.img_size=(401,512)
            model.patch_embed.num_patches = (401// 16) * (512 // 16)
            model.patch_embed=PatchEmbed((401,512),16,2,model.embed_dim)
            model.pos_embed = nn.Parameter(torch.zeros(1, model.patch_embed.num_patches + 1, model.embed_dim))
        elif args.model == 'effnetv2':
            model = effnetv2_s(num_classes=10)
        model.cuda()    
        # 是否ifft
        if args.ifft == 'False':
            dataset = myDataset
        else:
            dataset = myDataset_ifft
    elif args.channel == 4:
        dataset = myDataset_4
        model = DPAAlexNet_4channel(num_classes=num_classes)
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
