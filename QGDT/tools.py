# -*- coding: utf-8 -*-
"""
util主要完成了以下工作：
1.训练参数RNN
2.日志、当前路径、相关度计算函数
3.字典、语料库、RNNLM、拓展配置、全局配置
"""

__title__ = 'QGDT-cpu'
__author__ = 'Ex_treme'
__license__ = 'MIT'
__copyright__ = 'Copyright 2018, Ex_treme'

import logging
import os

import torch
import torch.nn as nn

from math import exp,pi

chinese = "[\u4e00-\u9fa5]"

DEBUG = True

# Device configuration
device = 'cpu'

# Hyper-parameters
embed_size = 128
hidden_size = 1024
num_layers = 2
num_epochs = 10
batch_size = 40  # 40
seq_length = 30  # 30
learning_rate = 0.002


def log(level, msg):
    global DEBUG
    if DEBUG:
        if level == 'info':
            logging.info(msg)
        elif level == 'warning':
            logging.warning(msg)
        elif level == 'debug':
            logging.debug(msg)
        elif level == 'error':
            logging.error(msg)


def get_current_path(path):
    d = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(d, path)


# jaccard相关度
def jaccard(x1, x2):
    intersection = [i for i in x1 if i in x2]
    union = [i for i in x1 if i not in x2]
    union[0:0] = x2
    return float(len(intersection)) / len(union)

# 计算正太分布概率值
def normal_distribution(x):
    return exp(-((x-0.5)**2)/2)/(2*pi)**0.5

# 计算方差
def variance(x_list):
    avg = sum(x_list)/len(x_list)
    sos = sum([(i-avg)**2 for i in x_list])
    return sos/len(x_list)

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)


class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()

    def get_data(self, path, batch_size=20):
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

                    # Tokenize the file content
        ids = torch.LongTensor(tokens)
        token = 0
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        num_batches = ids.size(0) // batch_size
        ids = ids[:num_batches * batch_size]
        return ids.view(batch_size, -1)


# RNN based language model
class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.embed(x)

        # Forward propagate LSTM
        out, (h, c) = self.lstm(x, h)

        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.reshape(out.size(0) * out.size(1), out.size(2))

        # Decode hidden states of all time steps
        out = self.linear(out)
        return out, (h, c)


def extend_config(config, config_items):
    """
    We are handling config value setting like this for a cleaner api.
    Users just need to pass in a named param to this source and we can
    dynamically generate a config object for it.
    """
    for key, val in list(config_items.items()):
        if hasattr(config, key):
            setattr(config, key, val)

    return config


class Configuration(object):
    def __init__(self):
        """
        Modify any of these QGDT properties
        TODO: Have a separate Config extend this!
        """

        self.LOG_ENABLE = True  # 是否开启日志
        self.LOG_LEVEL = 'INFO'  # 默认日志等级
        self.SVM_PATH = get_current_path('models/svm')  # SVM默认路径
        self.W2V_PATH = get_current_path('models/w2v')  # Word2Vec默认路径
        self.RNN_PATH = get_current_path('models/rnn')  # RNNLM默认路径
        self.DICT_PATH = get_current_path('models/rnn_dict')  # 字典默认路径
        self.RANDOM_SAMPLE = False  # 概率随机抽样
        self.LAMBDA = 0.4  # 融合因子
        self.ALPHA = 1.2  # 相似度计算调节因子
        self.BETA = 0.05  # 频度计算调节因子


if __name__ == '__main__':
    pass