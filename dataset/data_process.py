import numpy as np
from IPython.core.debugger import set_trace
from torchvision import transforms
from timm.data import create_transform
import matplotlib.pyplot as plt

def pre_process(x):
    x=np.log10(1+np.abs(np.fft.ifftshift(np.fft.ifft(x.T))).astype(np.float32))
    # x -= np.mean(x)
    # x /= np.std(x)
    # x=x[:,100:300]
    # plt.imshow(x.astype('uint8'))
    # plt.show()
    # trans_list = [
    #     transforms.ToTensor()
    #     # transforms.Resize(512,InterpolationMode='bicubic'),
    #     transforms.CenterCrop((512,201)), 
    #     # transforms.RandomHorizontalFlip(p=0.5),
    #     # transforms.RandomHorizontalFlip(p=0.5),
    #     ]
    # trans = transforms.Compose(trans_list)
    # x=trans(x)
    # plt.imshow(x.astype('uint8'))
    # plt.show()
    # set_trace()
    # transforms.
    # print(x_process.shape)
    return x