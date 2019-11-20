#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 20:47:10 2019

@author: daniel
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
# 加载模型
model = torch.load('bow.mdl')
print(model.named_parameters)

# 绘制出第二个全连接层的权重大小
#  model[2]即提取第二层, 网络一共4层，第0 层为线性神经元， 第1层为ReLU，第2层为第二层神经元连接，第3层为logsoftmax层

plt.figure(figsize = (10, 7))
for i in range(model[2].weight.size()[0]):
    weights = model[2].weight[i].data.numpy()
    plt.plot(weights, 'o-', label = i)

plt.legend()
plt.xlabel('Neuron in Hidden Layer')
plt.ylabel('Weights')
plt.savefig('Neuron_weights.jpg')
#print(model[0].weight.size())
#print(model[0].weight.data)
#print(model[2].weight.size())
#print(model[2].weight.data)

plt.figure(figsize = (10, 7))
for i in range(model[0].weight.size()[0]):
    #if i == 1:
        weights = model[0].weight[i].data.numpy()
        plt.plot(weights, alpha = 0.5, label = i)
plt.legend()
plt.xlabel('Neuron in Input Layer')
plt.ylabel('Weights')
plt.savefig('weights_0.jpg')

from txz_plot_function import model_weight_plot as tplot
tplot(model, layerid = 0)

# 加载词索引以及字典
import pickle
F = open('Data_to_load.pkl','rb')
Data_to_load = pickle.load(F)
F.close()

pos_sentences = Data_to_load['pos_sentences']
neg_sentences = Data_to_load['neg_sentences']
diction = Data_to_load['diction']

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
    
# 将第0层的各个圣经元与输入层的链接权重，挑出来最大的权重和最小的权重，并考察每个权重对应的单词是什么，把单词
# 打印出来
for i in range(len(model[0].weight)):
    print('\n')
    print('第{}个神经元'.format(i))
    print('max:')

    # 按 w来排序, 并记住这些w的索引号i
    st = sorted([(w,i) for i,w in enumerate(model[0].weight[i].data.numpy())])
    for i in range(1, 20):
        # list类型st[-i] 表示倒数第i个数
        word = index2word(st[-i][1],diction)
        print(word)
    print('min:')
    for i in range(20):
        word = index2word(st[i][1],diction)
        print(word)

"""
寻找错误的原因
"""
# 加载 测试数据集

import pickle
F = open('data.pkl', 'rb')
test = pickle.load(F)
F.close()
test_data, test_label = test['test_data'], test['test_label']

import pickle
Fs = open('sentences.pkl','rb')
sentences = pickle.load(Fs)
Fs.close()
print(sentences)

test_size = len(test_data)
print(test_size)

wrong_sentences = []
targets = []
j = 0
sent_indices = []
print("test_size:{}".format(test_size))
for data, target in zip(test_data, test_label):
    predictions = model(torch.tensor(data, dtype = torch.float).view(1, -1))
    #print(predictions.shape) # （1，2）
    # 按列计算出 一维predictions中最大值的下标
    pred = torch.max(predictions.data, 1)[1]
    # 打印可知道 torch.max(predictions.data, 1)[0] 表示数据内容，[1]才表示下标
    #print(torch.max(predictions.data, 1)[0])
    #print(pred.shape, pred.data)

    target = torch.tensor(np.array([target]), dtype = torch.long).view_as(pred)
    #print(target, pred)
    rights = pred.eq(target)
    # 示例, torch.Size([1]), <class 'torch.Tensor'>, [1]
    #print(rights.shape, type(rights), rights.data.numpy())

    # np.where(rights.numpy() == 0)为tuple类型，长度为1，所以[0] 为其元素, 
    # indices有时候是0长度的
    indices = np.where(rights.numpy() == 0)[0]
    #print(indices)
    for i in indices:
        wrong_sentences.append(data)
        targets.append(target)
        sent_indices.append(test_size + j + i)
    j += len(target)

idx = 1
#print(sent_indices, len(sent_indices))
#print(len(sentences), targets, len(targets))
#print(sentences)
#print(sentences[sent_indices[idx]], targets[idx].numpy())
lst = list(np.where(wrong_sentences[idx]>0)[0])
mm = list(map(lambda x:index2word(x, diction), lst))
print(mm)

# remained to do： 设计一个索引号，存储到dataset中，这样方便反向 求出错误的语句

#abc = model[0].weight.data.numpy().dot(wrong_sentences[idx].reshape(-1, 1))
#print(abc, abc.shape)


