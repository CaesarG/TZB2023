import scipy.io as sciio
import numpy as np
from PIL import Image
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    
    def __init__(self,class_num=10,frame_num=250,transform=None,target_transform=None):
        self.class_num=class_num
        self.frame_num=frame_num
        self.transform=transform
        self.target_transform=target_transform
        self.path_cwd=os.getcwd()
        
    def __getitem__(self,idx):
        # calculate index
        self.frame_idx=idx%self.frame_num+1
        self.class_idx=idx//self.frame_num+1
        self.load_data_EvEh(self.class_idx,self.frame_idx)
        image_Ev=torch.tensor(self.data_Ev_fft_abs)
        image_Eh=torch.tensor(self.data_Eh_fft_abs)
        label=self.class_idx
        if self.transform:
            image_Ev=self.transform(image_Ev)
            image_Eh=self.transform(image_Eh)
            
        return image_Ev,image_Eh,label
    
    def __len__(self):
        return self.class_num*self.frame_num

    def load_data_EvEh(self,class_idx,frame_idx):
        # get the name of mat file
        mat_name=self.path_cwd+'\\'+'DATA_02\\'+str(class_idx)+'\\'+'frame_'+str(frame_idx)+'.mat'
        mat_data=sciio.loadmat(mat_name)
        # self.data_Ev=np.abs(mat_data['frame_Ev'])
        # self.data_Eh=np.abs(mat_data['frame_Eh'])    
        self.data_Ev_fft_abs=np.abs(np.fft.ifftshift(np.fft.ifft(np.array(mat_data['frame_Ev'],dtype='complex').T)))
        self.data_Eh_fft_abs=np.abs(np.fft.ifftshift(np.fft.ifft(np.array(mat_data['frame_Eh'],dtype='complex').T)))
    
    def get_idx(self):
        return self.class_idx,self.frame_idx
    def print_idx(self):
        print("class idx is"+str(self.class_idx)+'\n')
        print("frame idx is"+str(self.frame_idx)+'\n')
    

# for class_idx in range(1,2):#11
#     for frame_idx in range(1,2):#251
        

        # print(data_Eh)

        
        # print(type(data_Ev))

if __name__ =='__main__':
    radar_data=CustomDataset()
    for i in range(2500):
        image_Ev,image_Eh,label=radar_data[i] #0-2499
        class_idx,frame_idx=radar_data.get_idx()

        plt.imsave('data2/'+str(class_idx)+'_'+str(frame_idx)+'_Ev.png', image_Ev)
        plt.imsave('data2/'+str(class_idx)+'_'+str(frame_idx)+'_Eh.png', image_Eh)
        # plt.title(label)
        # plt.imshow(image_Ev)
        # plt.show()
    