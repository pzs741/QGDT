# -*- coding: utf-8 -*-
"""
该模块完成了排序算法中频度计算部分，该模块依赖于RNNLM（RNN+LSTM语言模型）,
通过一句话中每个单词输入模型，输出层参数可以反映下一个词的频度，将下一个单词作为参数进入模型输入层，最终为问句打分。
"""
__title__ = 'QGDT-cpu'
__author__ = 'Ex_treme'
__license__ = 'MIT'
__copyright__ = 'Copyright 2018, Ex_treme'

import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))

from QGDT import Search2Frequency

if __name__ == "__main__":
    #初始化一个实例，传入语言模型和词典
    s = Search2Frequency(['Anti-DDoS流量清洗',' 查询周防护统计情况','功能介绍'],'models/rnn','models/rnn_dict')
    #计算搜索词序列的频度
    s.frequency_calculate()