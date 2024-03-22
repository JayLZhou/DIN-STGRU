# Video Prediction Models
旨在提供快速验证模型结果框架，本框架提供`ConvLSTM`，`ConvGRU`等`ConvRNN`结构的模型，也提供`Seq2Seq`的结构模型。  

目前支持的数据集有：
1. **卫星云图外推（FY-4A）**
2. **MovingMnist**

## 配置新模型

### 添加模型：

1. 在`core/layers`中添加需要的`cell`。  
2. 在`core/model_factory`中编写模型名字字典，如：


``` python
        networks_map = {
            'PhyDNet':
            PhyDNet,
            'RPNet':
            predict.RPNet,
        }
```

3. 在options中添加配置文件如`options/PhyDNet_mm.yml`
``` yml
dataset_name: "moving_mnist"
data_root: data/moving_mnist
save_dir: checkpoints/PhyDNet
tb_log_dir: tb_log/PhyDNet/
save_root: results/PhyDNet/
input_length: 10
total_length: 20
img_width: 64
img_height: 64
img_channel: 1
num_workers: 4

# padding_H: 4  # 如果想Padding，需要是偶数，因为这是上下Padding 2，就是padding_H=4
# padding_W: 0  # 默认不会Padding

# model
model_name: PhyDNet
pretrained_model: ~  # pretrained的模型
strict: True  # strict需要严格一致，如果模型内的参数发生小变化可以调成False

# gru:
num_hidden: [128, 128, 64]  # GRU的hidden
filter_size: 3  # 统一的kernel_size
stride: 1
patch_size: 1  # 如果有encode就不需要reshape patch
layer_norm: 1

# phy:
phy_num_hidden: [49]
phy_filter_size: 7

# 默认是没有encode输入的，只有PhyDNet需要encoder
# encoder:
encode_input: True
encoder:
  name: dcgan
  skip_connect: True
  context_channel: 64

# scheduled sampling
scheduled_sampling: 1 # 是否开启scheduled_sampling
sampling_stop_iter: 4000 # 停止scheduled_sampling的iter
sampling_start_value: 1.0 # 一般就是1
sampling_changing_rate: 0.00025 # 一般sampling_start_value/sampling_stop_iter

# optimization
lr: 0.0001  # 初始学习率
max_iterations: 30000  # 总的iter数
lr_steps: [2000, 3000, 4000, 5000, 6000] # 这个是是调节学习率的，2000过后乘0.5
reverse_input: 1 # 是否把序列正向输入后再反向输入
test_batch_size: 10 # test的batch大小 默认1
batch_size: 32 # train的batch大小
epochs: 100 # 总轮数
display_interval: 50 # 控制台输出间隔
test_interval: 800 # 测试间隔
gpu_ids: [1]  # 目前不支持多显卡，只能是指定哪张卡
```


## Train
```
python train_with_opt_pad.py -opt options/s2sconvgru.yml --is_training 1
```


## Test
配置好pretrained_model可以直接用来测试
```
python train_with_opt_pad.py -opt options/s2sconvgru.yml --is_training 0
```


## 配置参数详解

`dataset_name`: 就是数据集名字，没啥作用  

`data_root`: 数据集位置，`moving_mnist`就在`data/moving_mnist`  

`save_dir`: checkpoints存放位置  

`tb_log_dir`: tensorboard日志的存放位置  

`save_root`: test的时候结果的存放位置  

`input_length`: 输入序列长度  

`total_length`: 输入序列总长度  

`img_height`: 图片高度  

`img_width`: 图片宽度  

`img_channel`: 图片通道数，灰度图为1

`num_workers`: dataloader需要的num_workers，加速IO  

`padding_H`: H方向的Padding，需要是偶数，因为这是上下Padding 2，就是padding_H=4。默认不会Padding  

`padding_W`: 同`padding_H`  

`model_name`: 对应model_factory的名字  

`pretrained_model`: pretrained的模型，不提供请使用`~`  

`strict`: strict需要加载模型与模型定义严格一致，如果模型内的参数发生小变化可以调成False   

`num_hidden`: GRU等模型的hidden  

`filter_size`: 统一的kernel_size  

`stride`: 统一的stride  

`patch_size`: 如果有encode就不需要reshape patch，如果不需要reshape patch就置为1，不提供也是不会reshape的  

`layer_norm`: 模型中是否需要`layer_norm`

`phy_num_hidden`: PhyDNet特有的，PhyCell的num_hidden  

`phy_filter_size`: PhyDNet特有的，PhyCell的kernel_size

`scheduled_sampling`: 是否开启scheduled_sampling  
`sampling_stop_iter`: 停止scheduled_sampling的iter
`sampling_start_value`: scheduled_sampling的初始值，一般就是1  
`sampling_changing_rate`: 一般`sampling_start_value/sampling_stop_iter`  

`lr`: 初始学习率  
`max_iterations`: 总的iter数  
`lr_steps`: [2000, 3000, 4000, 5000, 6000] 这个是是调节学习率的，2000过后乘0.5  
`reverse_input`: 是否把序列正向输入后再反向输入
`test_batch_size`: test的batch大小 默认1
`batch_size`: train的batch大小
`epochs`: 总轮数
`display_interval`: 控制台输出间隔
`test_interval`: 测试间隔
`gpu_ids`: 目前不支持多显卡，只能是指定哪张卡`[0]`

## P.S.
默认是没有encode输入的，只有PhyDNet需要encoder
```yml
encoder:  
encode_input: True  
encoder:  
  name: dcgan  
  skip_connect: True  
  context_channel: 64  
```

