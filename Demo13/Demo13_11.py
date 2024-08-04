import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
pretrained_net = torchvision.models.resnet18(weights="IMAGENET1K_V1")
net = nn.Sequential(*list(pretrained_net.children())[:-2])
print(list(net.children()))
# VOC的类别
num_class = 21
net.add_module('fine_conv',nn.Conv2d(512, num_class, kernel_size=1))
# 这些参数保证了可能将fince_conv输出的10*15的特征转换为320*480
net.add_module('transpose_conv', nn.ConvTranspose2d(num_class, num_class, kernel_size=64,
                                                    padding=16, stride=32))


# 采用双线性插值初始化转置卷积层
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight

# 初始化转置卷积层
W = bilinear_kernel(num_class, num_class, 64)
net.transpose_conv.weight.data.copy_(W)
batch_size, crop_size = 32, (32, 480)
train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)


def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)


num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
d2l.plt.show()

