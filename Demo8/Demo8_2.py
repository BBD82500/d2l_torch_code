import collections
import re
from d2l import torch as d2l


d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  #@save
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# 文本总行数: {len(lines)}')
print(lines[0])
print(lines[10])

def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        # 这是python语法，因为字符串在 Python 中是可迭代对象，迭代字符串会逐字符返回它们，所以会将每个字符作为列表元素
        return [list(line) for line in lines]
    else:
        print("错误：未知词元类型：" + token)


tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])


class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """这里min_freq是针对可能对出现频率低的词丢掉，reserved_tokens是一些特别的词"""
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        # 这里返回就不是字典了，而是列表，其元素是元组[("a",1)...]
        self._token_freqs = sorted(counter.items(), key= lambda x:x[1], reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                # 默认判断的是键
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """获取某个词元的索引"""
        if not isinstance(tokens, (list, tuple)):
            # 用get方法来获取词元，如果没有就返回设置的self.unk
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """根据索引获得词元"""
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        # return [self.to_tokens(index) for index in indices]
        return [self.idx_to_token(index) for index in indices]

    def unk(self): # 未知词元索引为0
        return 0

    def token_freqs(self):
        return self._token_freqs


def count_corpus(tokens):
    """统计词元出现的频率"""
    # 这里的tokens是一维列表或二维列表,这里后面的or是判断二位列表的
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表层展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

# vocab = Vocab(tokens)
# print(list(vocab.token_to_idx.items())[:10])
#
# for i in [0,10]:
#     print('文本', tokens[i])
#     print('索引', vocab[tokens[i]])


def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 因为时光机器数据集不是每个文本行不一定都是一个句子
    # 或一个段落，所以将所有文本行展平到一个列表
    corpus = [vocab[token] for line in lines for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


corpus, vocab = load_corpus_time_machine()
print(len(corpus), len(vocab ))




