import torch
import torchvision
from torch import nn
from d2l import torch as d2l

d2l.set_figsize()
# 内容图片
content_img = d2l.Image.open("rainier.jpg")
# content_img = d2l.Image.open("content.jpg")
# d2l.plt.imshow(content_img)
# 风格图片
style_img = d2l.Image.open("autumn-oak.jpg")
# style_img = d2l.Image.open("style.jpg")
# d2l.plt.imshow(style_img)
# d2l.plt.show()

# 定于图像预处理和后处理
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])


def preprocess(img, image_shape):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)]
    )
    return transforms(img).unsqueeze(0)


def postprocess(img):
    img = img[0].to(rgb_std.device)
    # clamp限制输入值在[0,1],这里改变顺序是为了能够同时处理三个维度
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    # 再次将图像的通道顺序从 [H, W, C] 转换回 [C, H, W]，因为 ToPILImage 期望的是这种格式
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))
    # 给出代码理解：转换后每个块的每个行代表三个通道同一位置上的值，这样保证每次乘的时候，都是直接算之前三个通道同一位置上的值
    # a = torch.arange(16).reshape(4,4)
    # b = a.unsqueeze(0).repeat(3, 1, 1)
    # b = b.permute(1,2,0)
    # print(b.shape)
    # print(b)
    # mean = torch.tensor([1,2,3])
    # print(b * mean)


# 提取图像特征
pretrained_net = torchvision.models.vgg19(weights="DEFAULT")
style_layers, content_layers = [0, 5, 10, 19, 28], [25]
print(pretrained_net)
# 只保留要用的到VGG层
net = nn.Sequential(*[pretrained_net.features[i] for i in range(max(style_layers + content_layers) + 1)])


def extract_features(X, content_layers, style_layers):
    """逐层计算，并保留内容层和风格层的输出"""
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles


# 因为不需要训练VGG参数，可以在训练开始之前就获得内容和风格图像的内容特征和风格特征
def get_contents(image_shape, device):
    content_X = preprocess(content_img, image_shape).to(device)
    content_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, content_Y


def get_styles(img_shape, device):
    style_X = preprocess(style_img, img_shape).to(device)
    _, style_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, style_Y


# 定义损失函数：内容损失、风格损失、全变分损失

# 内容损失
def content_loss(Y_hat, Y):
    # 使用detach从动态计算梯度的树中分离目标，因为这里不要更新内容图片和风格图片
    # 这是一个规定的值，而不是有一个变量
    return torch.square(Y_hat - Y.detach()).mean()


# 风格损失
def gram(X):
    # 这里获得通道数和hw
    num_channels, n = X.shape[1], X.numel() // X.shape[1]
    X = X.reshape((num_channels, n))
    return torch.matmul(X, X.T) / (num_channels * n)


def style_loss(Y_hat, gram_Y):
    """这里用来gram_Y,是因为可以提前算出来Y的gram值"""
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()


# 全变分损失,避免生成图像有太多噪声
def tv_loss(Y_hat):
    """在图像处理中，边界像素通常会引入额外的处理复杂性，因为它们缺少相邻像素。例如，对于第0行，没有上一行来计算其与上一行的像素差异；
    对于第0列，没有左侧列来计算其与左侧列的像素差异。这种边界效应可能导致损失函数中的不连续性或者不准确的梯度估计。
    因此，为了避免处理这种边界效应，并且保证损失函数的计算能够尽可能准确地反映图像的平滑性，一般会从第1行和第1列开始计算相邻像素之间的差异，"""
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())


# 定义损失函数
content_weight, style_weight, tv_weight = 1, 1e4, 10


def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    # 分别计算内容损失、风格损失、全变分损失
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # 多所有损失求和
    l = sum(styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l


# 初始胡合成图像
class SynthesizedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        # nn.Parameter将一个张量封装成参数，使其能够方便地参与模型的优化和更新。
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight


def get_inits(X, device, lr, styles_Y):
    gen_img = SynthesizedImage(X.shape).to(device)
    gen_img.weight.data.copy_(X.data)
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer


# 训练模型
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    # 创建学习率调节器
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.9)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        trainer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_features(X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X))
            animator.add(epoch + 1, [float(sum(contents_l)), float(sum(styles_l)), float(tv_l)])
    return X


device, image_shape = d2l.try_gpu(), (300, 450)
net = net.to(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
# output = train(content_X, contents_Y, styles_Y, device, 0.3, 500, 50)
output = train(content_X, contents_Y, styles_Y, device, 0.3, 1000, 10)
d2l.plt.show()
