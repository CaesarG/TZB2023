import torch
import torch.nn as nn
import timm
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='RCS model Export')
    parser.add_argument('--resume', type=str, default='model_last_.pth', help='Weights resumed to export')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=10)

    model.conv_stem = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1), bias=False)

    checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
    print('loaded weights from {}, epoch {}'.format(args.resume, checkpoint['epoch']))
    state_dict_ = checkpoint['model_state_dict']
    model.load_state_dict(state_dict_, strict=False)
    model.to('cuda:0')
    model.eval()
    inputs = torch.randn(1, 2, 401, 512, dtype=torch.float32, device='cuda:0')
    traced_model = torch.jit.trace(model, inputs)
    torch.jit.save(traced_model, 'Tesla.dll')

    # torch.save(model, 'Tesla.dll')
