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

pos_sentences, neg_sentences, diction = Prepare_data(good_file, bad_file, True)

# 保存下pos_sentences, neg_sentences, diction
Data_to_load = {}
Data_to_load['pos_sentences'] = pos_sentences
Data_to_load['neg_sentences'] = neg_sentences
Data_to_load['diction'] = diction

import pickle
F = open('Data_to_load.pkl', 'wb')
pickle.dump(Data_to_load, F)
F.close()

# st为 字典 (出现数次数, 单词)的排序 tuple对的list
st = sorted([(v[1], w) for w,v in diction.items()])

# 输入一个句子和相应的词典，得到这个句子的向量化表示
# 向量的尺寸为词典中词汇的个数， i位置上面的数值为第i个单词在sentence中出现的频率

def sentence2vec(sentence, dictionary):
    vector = np.zeros(len(dictionary))
    for i in sentence:
        vector[i] += 1
    return (1.0*vector / len(sentence))

"""
 遍历所有的句子，将句子向量编码存到 dataset中， 情感（good 为0， bad为1存到 label中），划分数据集
"""

dataset = []
labels = []
sentences = [] # 存放原始句子

# 存下完整的句子
import pickle
F = open('sentences.pkl','wb')
pickle.dump(sentences, F)
F.close()

# 存正向评论,  遍历每一句话，将每句话的 词用字典中的index号替换，然后将其转为为词袋模型的向量 
for sentence in pos_sentences:
    new_sentence = []
    for i in sentence:
        if i in diction:
            new_sentence.append(word2index(i, diction))
    dataset.append(sentence2vec(new_sentence, diction))
    labels.append(0)
    sentences.append(sentence)

# 存反向评论，遍历每一句话，将每句话的 词用字典中的index号替换，然后将其转为为词袋模型的向量 
for sentence in neg_sentences:
    new_sentence = []
    for i in sentence:
        if i in diction:
            new_sentence.append(word2index(i, diction))
    dataset.append(sentence2vec(new_sentence, diction))
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

# 保存数据集
import pickle
dataset2save = {}
dataset2save['test_data'] = test_data
dataset2save['test_label'] = test_label
F = open('data.pkl', 'wb')
pickle.dump(dataset2save, F)
F.close()

"""
训练模型
"""
# 三层前馈神经网络，三层，第一层：线性层，加一个非线性层ReLU, 第二层线性层，第三层 LogSoftmax()层 中间有10个隐含层神经元
# 输入维度为字典的大小：每一段评论的词袋模型
model = nn.Sequential(
        nn.Linear(len(diction), 10),
        nn.ReLU(),
        nn.Linear(10,2),
        nn.LogSoftmax()
        )

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
          
# 损失函数为交叉熵
cost = torch.nn.NLLLoss()
# 优化算法为Adam， 可以自动调节学习率
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
records = []

# 循环10个Epoch
losses = []

for epoch in range(10):
    for i, data in enumerate(zip(train_data, train_label)):
        x, y = data
        x = torch.tensor(x, requires_grad = True, dtype = torch.float).view(1, -1)
        y = torch.tensor(np.array([y]), dtype = torch.long)        
        # 清空梯度
        optimizer.zero_grad()
        # 模型预测
        predict = model(x)
        # 计算损失函数
        loss = cost(predict, y)        
        # 将损失函数数值加入到列表中
        losses.append(loss.data.numpy())
        # 开始求梯度
        loss.backward()
        # 开始对参数进一步优化
        optimizer.step()
        
        # 每隔3000步， 抱一下验证集的数据， 输出临时结果

            
        if i % 3000 == 0:
            val_losses = []
            rights = []
            
            # 在所有的验证集上实验
            for j, val in enumerate(zip(valid_data, valid_label)):
                xx, yy = val
                xx = torch.tensor(xx, requires_grad = True, dtype = torch.float).view(1, -1)
                yy = torch.tensor(np.array([yy]), dtype = torch.long)
                predict_valid = model(xx)
                right = rightness(predict_valid, yy)
                rights.append(right)
                loss_valid = cost(predict_valid, yy)
                val_losses.append(loss.data.numpy())
            # 将验证集上的平均数据精确计算出来
            right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
            print('第{}轮, 训练损失:{:.2f}, 校验损失:{:.2f}, 校验准确率:{:.2f}'.format(epoch, np.mean(losses), np.mean(val_losses), right_ratio))
            
            records.append([np.mean(losses), np.mean(val_losses), right_ratio])

a = [i[0] for i in records]
b = [i[1] for i in records]
c = [i[2] for i in records]
plt.plot(a, label = 'Train Loss')
plt.plot(b, label = 'Valid Loss')
plt.plot(c, label = 'valid accuracy')
plt.xlabel('epoches')
plt.ylabel('loss & accuracy')
plt.legend()
plt.savefig('loss_&_accuracy.jpg')

# 保存模型
torch.save(model, 'bow.mdl')
# 加载模型 model = torch.load('bow.mdl')  