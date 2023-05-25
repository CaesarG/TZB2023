# Paper reading

Some notes after read related paper
# 1.Mbconv block and mobilenetV3
## Mbconv

Read this link for mbconv
https://blog.csdn.net/qq_41488595/article/details/123290023

It is similar to mobilenetV3

## MobileNeV1-V3

### MobileNetV1

### Depthwise convolutions
MobileNetV1 introduced depthwise separable convolutions as an efficient replacement for
traditional convolution layers. Depthwise separable convolutions effectively factorize traditional convolution by separating spatial filtering from the feature generation mechanism. Depthwise separable convolutions are defined by
two separate layers: light weight depthwise convolution for
spatial filtering and heavier 1x1 pointwise convolutions for
feature generation.

### MobileNetV2

### Linear bottlneck,Inverted residual structure
1x1 expansion Conv -> depthwise Conv ->1x1 projection with residual

### MobileNetV3
Some ateention blocks

swish + squeeze and excitation
1x1 expansion Conv -> depthwise Conv -> Squeeze+excitation with residual

### SENet
Squeeze and excitation

### 1-10
存放1~10个类别的rcs数据，每个文件夹存放250帧的mat文件，每帧包含Ev,Eh两种极化接收方式，两种接收数据大小均为401x512，401对应8G到10GHz的采样点数，
512对应不同方位角俯仰角的接收位置。
### annotations
存放训练集、验证集、测试集文件路径与类别的txt文件
## dataset
### annotation_gen.py
生成txt格式的标签文件
### observer.py
观察转换后数据情况
### dataset.py
直接abs接收数据的dataset，送入网络时是2x401x512
### dataset_4.py
分开使用接收数据实部虚部的dataset，送入网络时是4x401x512
### dataset_ifft.py
对频谱数据做ifft到时域后取abs的dataset，送入网络时是2x401x512
## model
存放Alexnet，VGG网络及各种trick.
### pretrain_weight
存放网络模型的与训练权重，加速网络训练速度，优化训练效果。Alexnet的预训练权值文件原链接丢失，目前方便获取的只有VGG权值。
## test_result
存放test的预测结果
## weight_of_model
存放训练结果，包括.pth文件和训练损失记录
## eval.py
准确率测试
## loss.py
损失函数，采用交叉熵损失函数
## test.py
生成预测结果
## train.py
训练文件
## main.py
主文件，train,eval,test均在通过主文件完成

# 2.使用说明
注：batch_size根据电脑配置和模型大小调整，一般取2的次幂数
## 2通道ifft训练指令：
Alexnet:<br>
python main.py --phase train --channel 2 --model Alexnet --ifft True --batch_size 16<br>
VGG:<br>
python main.py --phase train --channel 2 --model vgg --ifft True --batch_size 16
## 2通道ifft指标测试指令
Alexnet:<br>
python main.py --phase eval --channel 2 --model Alexnet --ifft True --batch_size 16 --resume path(比如model_24.pth) 
--test_txt test.txt<br>
VGG:<br>
python main.py --phase eval --channel 2 --model vgg --ifft True --batch_size 16 --resume path(比如model_24.pth) 
--test_txt test.txt

Autoformer:<br>
python main.py --phase train --channel 2 --model autoformer --ifft True --batch_size 16 --cfg ./experiments/supernet/supernet-T.yaml

Effcientnet:<br>
python main.py --phase train --channel 2 --model effcientnet --ifft True --batch_size 16 --cfg ./experiments/supernet/supernet-T.yaml
## 2通道abs训练指令：
python main.py --phase train --channel 2 --model Alexnet --ifft False --batch_size 16
## 2通道指标测试指令：
python main.py --phase eval --channel 2 --model Alexnet --ifft False --batch_size 16 --resume path(比如model_24.pth) 
--test_txt test.txt
## 4通道训练指令：
python main.py --phase train --channel 4 --model Alexnet --ifft False --batch_size 8
## 4通道指标测试指令：
python main.py --phase eval --channel 4 --model Alexnet --ifft False --batch_size 8 --resume path(比如model_24.pth) 
--test_txt test.txt