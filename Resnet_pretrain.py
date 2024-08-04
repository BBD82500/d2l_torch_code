import torch
from torch import nn
from d2l import torch as d2l
import torchvision
# 使用预训练的模型进行识别，可以达到90准确率
# 进行变换
# 使用RGB通道的均值和方差，以标准化每个通道（这是因为ImageNet也使用了，所以都应该使用）
# 如果这里不用，应该要引入BN
from torch.utils.data import DataLoader

normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_augs = torchvision.transforms.Compose([
    # 随机裁剪图片，并将裁剪后的图片缩放为224X224
    torchvision.transforms.RandomResizedCrop(224),
    # 进行随机水平翻转
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize
])
test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256, 256]),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize
])
# 除了上面手动添加转换，也可以使用torch自带的
weights = torchvision.models.ResNet18_Weights.DEFAULT
preprocess = weights.transforms()

batch_size = 128
# 读取参数
train_set = torchvision.datasets.CIFAR10("./CIFAR10", train=True, transform=preprocess, download=True)
test_set = torchvision.datasets.CIFAR10("./CIFAR10", train=False, transform=preprocess, download=True)
train_iter = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4)
test_iter = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=4)


# 读取imagenet的参数进行微调CFAIR10
# 一样的
# pretrained_net = torchvision.models.resnet18(weights="IMAGENET1K_V1")weights="DEFAULT"
pretrained_net = torchvision.models.resnet18(weights="DEFAULT")
# 这里Resnet18前面的特征层不改变，但是后面的输出层改变，因为这里我们使用CIFAR10是输出10个
pretrained_net.fc = nn.Linear(pretrained_net.fc.in_features, 10)
nn.init.xavier_uniform_(pretrained_net.fc.weight)


# 训练
# ----------注意：一定要关闭训练模型里面的初始化-------------------
def train_fine_tuning(net, train_iter, test_iter, num_epochs, learning_rate, device, param_group=True):
    """Train a model with a GPU (defined in Chapter 6).

    Defined in :numref:`sec_lenet`,注意这里param_group代表是否用预训练的参数"""

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    # net.apply(init_weights)
    print('training on', device)
    net.to(device)
    if param_group:
        # 这里是进行学习率的修改，对已经训练好的特征层学习率小一点，重新定义的输出层的学习率为其10倍
        params_1x = [param for name, param in net.named_parameters()
                     if name not in ["fc.weight", "fc.bias"]]
        optimizer = torch.optim.SGD([{'params': params_1x},
                                      {'params': net.fc.parameters(),
                                       'lr': learning_rate * 10}],
                                     lr=learning_rate, weight_decay=0.001)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    loss = nn.CrossEntropyLoss(reduction="none") # 引入了NONe，所以下面的l要用l.sum().backward()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.sum().backward()
            optimizer.step()
            with torch.no_grad():
                # 这里也修改为了l.sum()
                metric.add(l.sum() * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
        print(epoch)
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

train_fine_tuning(pretrained_net, train_iter, test_iter, 5, 5e-5, d2l.try_gpu())
d2l.plt.show()