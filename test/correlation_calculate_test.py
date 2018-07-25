# -*- coding: utf-8 -*-
"""
该模块对查询词进行判定，舍弃冗余（对生成问句累赘）查询词，输出最终检索词。
"""

__title__ = 'QGDT-cpu'
__author__ = 'Ex_treme'
__license__ = 'MIT'
__copyright__ = 'Copyright 2018, Ex_treme'

import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))

from QGDT import Terms2Search

if __name__ == "__main__":
    #初始化一个实例
    t = Terms2Search(['Anti-DDoS流量清洗',' 攻击事件能否及时通知？'],'models/svm')
    #计算搜索序列的相关度
    t.correlation_calcuulate()