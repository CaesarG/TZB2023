import numpy as np
from IPython.core.debugger import set_trace
from timm.data import create_transform
import matplotlib.pyplot as plt
import torchvision.transforms as T

def pre_process(x,type=['imaging']):
    if 'abs_ifft' in type:
        x=np.log10(1+np.abs(np.fft.ifftshift(np.fft.ifft(x.T)))).astype(np.float32)
    elif 'imaging' in type:
        x=np.fft.ifft(x.T)
        x=np.fft.fftshift(np.log10(1+np.abs(np.fft.fft(x.T))).T).astype(np.float32)   
    elif 'imaging_log' in type:
        x=np.fft.ifft(x.T)
        x=np.fft.fftshift(np.log10(1+np.abs(np.fft.fft(x.T))).T).astype(np.float32)      
    elif 'only_log' in type:
        x=np.log10(1+np.abs(x.T)).astype(np.float32)
    if 'mean' in type:
        x -= np.mean(x)
        
    if 'half_pic' in type:
        x=x[:,100:300]
    # x=np.log10(1+np.abs(np.fft.fftshift(np.fft.fft(x)))).astype(np.float32)
    # x /= np.std(x)

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