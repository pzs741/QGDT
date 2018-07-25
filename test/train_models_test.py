# -*- coding: utf-8 -*-
"""
训练模块单元测试:
该模块独立，可以作为模型训练器单独使用，TrainSVM、TrainW2V、TrainRNN都继承TrainModel这个类，
初始化的参数都为：origin_path(原始数据集相对路径)、train_path（转换后的训练集保存路径）、model_path（模型保存路径）
注意：RNNLM初始化多一个dict_path（数据字典保存路径），以上参数若不传递，则默认使用默认路径。
"""
__title__ = 'QGDT-cpu'
__author__ = 'Ex_treme'
__license__ = 'MIT'
__copyright__ = 'Copyright 2018, Ex_treme'

import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))

from QGDT import TrainRNN,TrainSVM,TrainW2V

if __name__ == "__main__":
    # 初始化一个类
    t = TrainSVM()
    # 将原数据转换为训练集
    t.origin_to_train()
    # 开始训练（训练完毕后模型自动保存到默认路径）
    t.train()
