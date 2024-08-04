import math

import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# for X, Y in train_iter:
#     print(X)
#     print(Y)


# 将词元变为独热编码
X = torch.arange(10).reshape((2, 5))
print(X)
# 这里转置是为了能够直接获取某一个时间的整个序列，如X[0::]可以获得0时间步的所有词元，X[1::]获得时间步1
# print(F.one_hot(X.T, 28))

# 初始化模型参数，num_hiddens：隐藏单元数
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        # 正态分布（均值为0，标准差为1）的随机数
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        # 用类实例方法
        param.requires_grad_(True)
    return params


# 定义循环神经网络模型
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)


# 定义RNN
# 注意这里修改了inputs的维度，是在其最外层的维度实现循环，以便随时间步更新小批量数据的隐状态H
def rnn(inputs, state, params):
    # inputs形状为（时间步数，批量大小，词表大小）
    # 使用tanh做激活函数
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # x形状为(批量大小，词表大小)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    # 输出为：[时间步数*批量大小，词表大小]，最终的隐藏状态，形状为 (批量大小, 隐藏状态大小),(H,)确定返回的是元组
    return torch.cat(outputs, dim=0), (H,)


class RNNModelScratch:
    """从零实现的循环神经网络模型"""

    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


# num_hiddens = 512
# net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params, init_rnn_state, rnn)
# state = net.begin_state(X.shape[0], d2l.try_gpu())
# Y, new_state = net(X.to(d2l.try_gpu()), state)
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.to(d2l.try_gpu()), state)
print(Y.shape, len(new_state), new_state[0].shape)

def predict_ch8(prefix, num_preds, net, vocab, device):  # @save
    """在prefix后面生成新字符"""
    state = net.begin_state(batch_size=1, device=device)
    # 将输入字符转化为idx
    outputs = [vocab[prefix[0]]]
    # .reshape((1, 1))是为了把数据变成(1,1)的tensor
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # 预热期，用于更新H，即更新给定字符的H隐藏状态
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


print(predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu()))

def grad_clipping(net, theta):
    """截断梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

# 训练
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一轮"""
    state, timer, = None, d2l.Timer()
    metric = d2l.Accumulator(2) # 训练损失之和，词元数量
    for X, Y in train_iter:
        # 这里X,Y 形状相同，（批量大小，时间步数）
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0],device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state 对于nn.GRU是一个张量
                state.detach_()
            else:
                # state对于nn.LSTM或对于我们从零实现的模型（现在这个RNN）是一个由张量组成的元组
                for s in state:
                    s.detach_()
        # 代码讲解，这里将y先转置，则第二维上就是同一时间步的值即每行是一个时间步上每个批量的词元，
        # 然后再排成一列，形成了T0(样本1)，T0(样本2)....T1(样本1)这样的排序
        # 正好等于之前rnn函数输出的结果，即输出批量大小*时间步数行的输出
        # 然后再多目标匹配即可-->解释：因为y_hat形状为批量大小*时间步数，vocab_size，而y形状为批量大小*时间步数，就是多目标匹配
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            # 使用官方的优化器
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            # 使用自己的优化器,没有ero_grad()是在我们的里面写了
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

# 训练函数
def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_ranom_iter=False):
    """训练模型"""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device) # 预测50个字符
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device, use_ranom_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))

num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(),use_ranom_iter=True)
d2l.plt.show()


