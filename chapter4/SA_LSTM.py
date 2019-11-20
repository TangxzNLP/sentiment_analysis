#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 18:08:39 2019

@author: daniel
"""

"""
利用词袋模型初始化数据，然后进行训练
"""
import torch
import torch.nn as nn
import torch.optim
import numpy as np
import matplotlib.pyplot as plt

import re
import jieba
from collections import Counter
from bpe import multi_bpe, load_model
E = load_model()


# 数据处理来源

good_file = 'data/good.txt'
bad_file = 'data/bad.txt'

# 过滤文本中的符号
def filter_punc(sentence):
    sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\'“”《》?“]+|[+——！，。？、～~@#￥%……&*（）：]+", "", sentence)
    return sentence   

# 扫描所有的文本，分词，建立词典，分出正向与负向的评论， is_filter是否要过滤调标点符号

def Prepare_data(good_file, bad_file, is_filter = True):
    # 存储所有的词
    all_words = []
    pos_sentences = [] # 存储正向的评论
    neg_sentences = [] # 存储负向的评论
    with open(good_file, 'r') as fr:
        for idx, line in enumerate(fr):
            if is_filter:
                line = filter_punc(line)
            words = jieba.lcut(line)
            if len(words) > 0:
                all_words += words
                pos_sentences.append(words)
    print('{0}包含{1}行,{2}个词'.format(good_file, idx + 1, len(all_words)))
    
    count = len(all_words)
    with open(bad_file, 'r') as fr:
        for idx, line in enumerate(fr):
            if is_filter:
                line = filter_punc(line)
            words = jieba.lcut(line)
            if len(words) > 0:
                all_words += words
                neg_sentences.append(words)
    print('{0}包含{1}行,{2}个词'.format(bad_file, idx + 1, len(all_words)-count))
    
    # 使用all_words建立词典， diction的每一项为{w:[id,单词出现的次数]}
    diction = {}
    cnt = Counter(all_words) # Counter() 为词频统计
    for word, freq in cnt.items():
        diction[word] = [len(diction), freq]
    print('字典大小:{}'.format(len(diction)))
    
    return (pos_sentences, neg_sentences, diction)


"""
 建立词->index 以及 index->词 的字典
"""

# 根据单词返还单词的编码
    
def word2index(word, diction):
    if word in diction:
        value = diction[word][0]
    else:
        value = -1
    return value

# 根据编码获得单词
def index2word(index, diction):
    for w,v in diction.items():
        if v[0] == index:
            return w
    return None

pos_sentences, neg_sentences, diction = Prepare_data(good_file, bad_file, False)


"""
 遍历所有的句子，将句子向量编码存到 dataset中， 情感（good 为0， bad为1存到 label中），划分数据集
"""

dataset = []
labels = []
sentences = [] # 存放原始句子

# 存正向评论,  遍历每一句话，将每句话的 词用字典中的index号替换，然后将其转为为词袋模型的向量 
for sentence in pos_sentences:
    new_sentence = []
    for i in sentence:
        if i in diction:
            new_sentence.append(word2index(i, diction))
    dataset.append(new_sentence)
    labels.append(0)
    sentences.append(sentence)

# 存反向评论，遍历每一句话，将每句话的 词用字典中的index号替换，然后将其转为为词袋模型的向量 
for sentence in neg_sentences:
    new_sentence = []
    for i in sentence:
        if i in diction:
            new_sentence.append(word2index(i, diction))
    dataset.append(new_sentence)
    labels.append(1)
    sentences.append(sentence)

# 打乱所有的索引号次序, 使用 np.random.permutation()函数
indices = np.random.permutation(len(dataset))

# 根据打乱的顺序重新生成新的的dataset, labels以及对应的sentences

dataset = [dataset[i] for i in indices]
labels = [labels[i] for i in indices]
sentences = [sentences[i] for i in indices]

# 对数据进行划分，分为训练集， 验证集与测试集合， 其中，验证集与测试集各占1/10
test_size = len(dataset)//10

train_data = dataset[2*test_size:]
train_label = labels[2*test_size:]

test_data = dataset[:test_size]
test_label = labels[:test_size]

valid_data = dataset[test_size:2*test_size]
valid_label = labels[test_size:2*test_size]

print(test_data)


"""
训练模型
"""
# 计算预测错误率的函数, predictions是模型给出的一组预测结果，batch_size行 num_classes列的矩阵,
# labels是数据中正确的答案
def rightness(predictions, labels):
    # 每行表示一个预测，其中最大概率的是预测的结果，predictions.data是取tensor类型的predictions的数据, 
    # 1表示第一个维度，也就是列。 pred[1]表示数据的下标,
    # 由于最后一层设置为2，故下标只能为0 或者 1
    pred = torch.max(predictions.data, 1)[1]
    # 判断下标是否和label值相等, pred.eq(..) 并求和
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels) # 返回正确的数量以及这次一共比较了多少元素


"""
数学公式：参考公式 https://blog.csdn.net/david0611/article/details/81090294
i<t> = sigmoid(Wii * x<t> + bii + Whi * h<t-1> + bhi)
f<t> = sigmoid(Wif * x<t> + bif + Whf * h<t-1> + bhf)
g<t> = tanh(Wig * x<t> + big + Whc * h<t-1> + bhg)
o<t> = sigmoid(Wio * x<t> + bio + Who * h<t-1> + bho)
c<t> = f<t> * c<t-1> + i<t> * g<t>
h<t> = o<t> * tanh(c<t>)

"""

class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMNetwork, self).__init__()
        self.hidden_size = hidden_size
        # 一个embedding层
        self.embed = nn.Embedding(input_size, hidden_size)
        self.ft = nn.Linear(2*hidden_size, hidden_size)
        self.it = nn.Linear(2*hidden_size, hidden_size)
        self.gt = nn.Linear(2*hidden_size, hidden_size)
        self.ot = nn.Linear(2*hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden=None):
        
        #input尺寸: seq_length
        #词向量嵌入
        x = self.embed(input)
        #embedded尺寸: seq_length, hidden_size
        hidden_h = hidden[0]
        hidden_c = hidden[1]
        combined = torch.cat((x, hidden_h), 1)

        # 计算forget层
        ft_hidden = self.ft(combined)
        ft_hidden = torch.nn.functional.sigmoid(ft_hidden)

        # 计算input层
        it_hidden = self.it(combined)
        it_hidden = torch.nn.functional.sigmoid(it_hidden)

        # 计算gt 层
        gt_hidden = self.gt(combined)
        gt_hidden = torch.nn.functional.tanh(gt_hidden)

        # 计算ot层
        ot_hidden = self.ot(combined)
        ot_hidden = torch.nn.functional.sigmoid(ot_hidden)

        # 更新隐含状态，计算output层

        hidden_c = ft_hidden * hidden_c + it_hidden * gt_hidden
        hidden_h = ot_hidden * torch.nn.functional.tanh(gt_hidden)
        hidden = (hidden_h, hidden_c)

        output = self.fc(hidden_h)
        output = self.softmax(output)
        
        return output,hidden
    
    def initHidden(self):
        hidden_h = torch.zeros(1, self.hidden_size)
        hidden_c = torch.zeros(1, self.hidden_size)
        return (hidden_h, hidden_c)
    
# 开始训练这个LSTM，10个隐含层单元
lstm = LSTMNetwork(len(diction), 10, 2)

# 交叉熵评价函数
cost = torch.nn.NLLLoss()

# Adam优化器
optimizer = torch.optim.Adam(lstm.parameters(), lr = 0.001)
records = []

# 学习周期10次
losses = []
for epoch in range(10):
    
    for i, data in enumerate(zip(train_data, train_label)):
        x, y = data
        # x shape->(n, 1)
        x = torch.tensor(x, dtype = torch.long).unsqueeze(1)
        # print(x.shape, x)
        #x尺寸：seq_length（序列的长度）
        y = torch.tensor(np.array([y]), dtype = torch.long)
        #x尺寸：batch_size = 1,1
        optimizer.zero_grad()
        
        #初始化隐含层单元全为0
        hidden = lstm.initHidden()
        # hidden尺寸：batch_size = 1, hidden_size
        
        #手动实现LSTM的时间步循环，x的长度就是总的循环时间步，因为要把x中的输入句子全部读取完毕
        # 该模型为多对1模型，所以output每次被覆盖
        for s in range(x.size()[0]):
            output, hidden = lstm(x[s], hidden)
        
        #校验函数
        # print(output.shape)
        loss = cost(output, y)
        losses.append(loss.data.numpy())
        loss.backward()
        # 开始优化
        optimizer.step()


        if i % 3000 == 0:
            # 每间隔3000步来一次校验集上面的计算
            val_losses = []
            rights = []
            for j, val in enumerate(zip(valid_data, valid_label)):
                x, y = val
                x = torch.tensor(x, dtype = torch.long).unsqueeze(1)
                y = torch.tensor(np.array([y]), dtype = torch.long)
                hidden = lstm.initHidden()
                for s in range(x.size()[0]):
                    output, hidden = lstm(x[s], hidden)
                right = rightness(output, y)
                rights.append(right)
                loss = cost(output, y)
                val_losses.append(loss.data.numpy())
            # 计算准确度
            right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
            print('第{}轮，训练损失：{:.2f}, 测试损失：{:.2f}, 测试准确率: {:.2f}'.format(epoch, np.mean(losses),
                                                                        np.mean(val_losses), right_ratio))
            records.append([np.mean(losses), np.mean(val_losses), right_ratio])

# 绘制误差曲线
a = [i[0] for i in records]
b = [i[1] for i in records]
c = [i[2] for i in records]
plt.plot(a, label = 'Train Loss')
plt.plot(b, label = 'Valid Loss')
plt.plot(c, label = 'Valid Accuracy')
plt.xlabel('Steps')
plt.ylabel('Loss & Accuracy')
plt.legend()
plt.savefig('LSTM_Hand.jpg')

# 保存模型
torch.save(lstm, 'lstm_Hand.mdl')