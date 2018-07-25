# -*- coding: utf-8 -*-
"""
该模块完成了排序算法中相似度计算部分，该模块依赖于W2V词向量模型，使用WDM算法计算查询词与预定义模板的匹配度。
"""

__title__ = 'QGDT-cpu'
__author__ = 'Ex_treme'
__license__ = 'MIT'
__copyright__ = 'Copyright 2018, Ex_treme'

import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))

from QGDT import Search2Similar

if __name__ == "__main__":
    #初始化一个实例
    s = Search2Similar(['Anti-DDoS流量清洗',' 查询周防护统计情况','功能介绍'],'models/w2v')
    #计算搜索词序列与预定义模板集的相似度
    s.similarity_calculate()

