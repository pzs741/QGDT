# -*- coding: utf-8 -*-
"""
该模块将相关度判定模块、相似度计算模块、频度计算模块融合在一起，
通过精心设计的信息检索模型为生成问句排序打分，最后由得分最高的模板生成相应问句。
"""

__title__ = 'QGDT-cpu'
__author__ = 'Ex_treme'
__license__ = 'MIT'
__copyright__ = 'Copyright 2018, Ex_treme'

import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))

from QGDT import QG

if __name__ == "__main__":
    #相似度序列
    sim_list = [2.261679196747406, 3.075403727118124, 2.871388395024239, 2.2647689008194836]
    #频度序列
    fre_list = [0.3126591742038727, 0.0, -0.19479990005493164, -0.19479990005493164]
    #初始化实例
    q = QG(sim_list,fre_list,0.2,0.3,0.5)
    #排序打分
    q.ranking()
