import math
import os
import random
import torch
from d2l import torch as d2l


d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
                       '319d85e578af0cdc590547f26231e4e31cdf1e42')


def read_ptb():
    """将PTB数据集加载到文本行的列表中"""
    data_dir = d2l.download_extract('ptb')
    # Readthetrainingset.
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    # 返回每个单词
    return [line.split() for line in raw_text.split('\n')]


sentences = read_ptb()
print(f'# 句子数::{len(sentences)}')
vocab = d2l.Vocab(sentences, min_freq=10)
print(f'vocab size: {len(vocab)}')

# 进行下采样
def subsample(sentences, vocab):
    """下采样高频词"""
    # 排除未知词元<unk>
    sentences = [[token for token in line if vocab[token] != vocab.unk] for line in sentences]
    counter = d2l.count_corpus(sentences)
    num_tokens = sum(counter.values())
    # 如果在下采样期间保留词元，则返回True
    def keep(token):
        return (random.uniform(0, 1)) < \
               math.sqrt(1e-4 / counter[token] * num_tokens)
    return [[token for token in line if keep(token)] for line in sentences], counter

subsampled, counter = subsample(sentences, vocab)

d2l.show_list_len_pair_hist(
    ['origin', 'subsampled'], '# tokens per sentence',
    'count', sentences, subsampled)
d2l.plt.show()

def compare_counts(token):
    return (f'"{token}"的数量：'
            f'之前={sum([l.count(token) for l in sentences])}, '
            f'之后={sum([l.count(token) for l in subsampled])}')

print(compare_counts('the'))
print(compare_counts('join'))

corpus = [vocab[line] for line in subsampled]
print(corpus[:3])

# 进行中心词和上下文词的提取
def get_centers_and_contexts(corpus, max_window_size):
    """返回跳元模型的中心词和上下文词"""
    centers, contexts = [], []
    for line in corpus:
        # 要形成中心词-上下文词对，每个句子至少要两个词
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)): # 上下文窗口中间i
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size), min(len(line), i + 1 + window_size)))
            # 从上下文词中剔除中心词
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts


tiny_dataset = [list(range(7)), list(range(7, 10))]
print('数据集', tiny_dataset)
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('中心词', center, '的上下文词是', context)

all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
print("中心词数：", len(all_centers))
print(f'# “中心词-上下文词对”的数量: {sum([len(contexts) for contexts in all_contexts])}')


# 负采样
class RandomGenerator:
    """根据n个采样权重在{1,...,n}中随机抽取"""
    def __init__(self, sampling_weights):
        # 排除
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            # 缓存k个随机采样结果
            self.candidates = random.choices(self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i-1]


generator = RandomGenerator([2, 3, 4])
print([generator.draw() for _ in range(10)])

# 进行负采样
def get_negatives(all_contexts, vocab, counter, K):
    """返回负采样的噪声词：注意这里书上文字是5个，但是代码是五倍，对其进行修改"""
    # 索引为1、2、...（索引0是词表中排除的未知标记）
    sampling_weights = [counter[vocab.to_tokens(i)] ** 0.75
                        for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) + K:
            neg = generator.draw()
            # 噪声词不能是上下文词
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives


all_negatives = get_negatives(all_contexts, vocab, counter, 5)

def batchify(data):
    """返回带有负采样的跳远模型的小批量样本"""
    max_len = max(len(c) + len(n) for _, c, n in data) # 生成器表达式，用来减少开销
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += \
            [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (torch.tensor(centers).reshape((-1, 1)), torch.tensor(contexts_negatives),
                torch.tensor(masks), torch.tensor(labels))


x_1 = (1, [2, 2], [3, 3, 3, 3])
x_2 = (1, [2, 2, 2], [3, 3])
batch = batchify((x_1, x_2))

names = ['centers', 'contexts_negatives', 'masks', 'labels']
for name, data in zip(names, batch):
    print(name, '=', data)

# 整合
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """下载ptb数据集，然后加载到内存"""
    num_workers = 0
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)

    class PTBDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives
        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index], self.negatives[index])
        def __len__(self):
            return len(self.centers)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)
    data_iter = torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True, collate_fn=batchify, num_workers=num_workers
    )
    return data_iter, vocab


data_iter, vocab = load_data_ptb(512, 5, 5)
for batch in data_iter:
    for name, data in zip(names, batch):
        print(name, 'shape:', data.shape)
    break

