import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# 定义模型
# 构造循环神经网络层 256个隐藏单元
num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)
# 初始化状态
# 形状(隐藏层大小，批量大小，隐藏元单元)
state = torch.zeros((1, batch_size, num_hiddens))
print(state.shape)
# 注意torch的隐藏层输出是隐藏层的输出，不是前面的输出
X = torch.rand(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
print(Y.shape, state_new.shape)

# 定义RNNModel类
class RNNModel(nn.Module):
    """循环神经网络模型"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果rnn是双向的，num_directions 应该是2，否则是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            # 如前面所说，手动构造一个输出层
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 注意RNN输出的y形状是（时间步数，批量大小，隐藏单元数）
        # 全连接层首先将Y的形状改为(时间步数X批量大小，隐藏单元数量)
        # 它的输出层形状为(时间步数X批量大小，词表大小)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态或者是RNN
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))


device = d2l.try_gpu()
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
print(d2l.predict_ch8('time traveller', 10, net, vocab, device))

num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)
d2l.plt.show()