import timm 
import torch
from urllib.request import urlopen
from PIL import Image
import timm
from huggingface_hub import login
from huggingface_hub import hf_hub_download
import torch.nn as nn
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
# login(token='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')
# all_model=timm.list_models()
# print(all_model)
from timm.models.byoanet import halo2botnet50ts_256
# model = timm.create_model("hf_hub:timm/botnet26t_256.c1_in1k", pretrained=True)
# model.stem.conv1.conv=nn.Conv2d(2,24,kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=False)
# print(model)
# model = AutoModelForImageClassification.from_pretrained("facebook/convnext-base-224")

model = timm.create_model("hf_hub:timm/convnext_base.fb_in22k_ft_in1k", pretrained=True)
# model = timm.create_model("hf_hub:timm/convnextv2_base.fcmae_ft_in22k_in1k", pretrained=True)
model.stem[0]=nn.Conv2d(2,128,(4,4),stride=(4,4))
print(model)