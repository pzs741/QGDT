# -*- coding: utf-8 -*-
"""
question_generate 主要完成了：
1.结合相似度和频度的排序算法
2.序号到模板的映射
3.问句生成
"""
__title__ = 'QGDT-cpu'
__author__ = 'Ex_treme'
__license__ = 'MIT'
__copyright__ = 'Copyright 2018, Ex_treme'

from math import e
from QGDT.tools import log


class QG(object):
    """问句生成模型-->排序打分"""

    def __init__(self,sim_list,fre_list,lamda,alpha,beta):
        """
        Keyword arguments:
        sim_list                -- 相似度序列，list类型
        fre_list                -- 频度序列，list类型
        len                     -- 序列长度，int类型
        lamda                   -- rankang系数，float类型,[0,1]
        alpha                   -- 相似度系数，float类型
        beta                    -- 频度系数，int类型
        """
        if sim_list.__len__() is not fre_list.__len__():
            raise Exception('检索模型初始化错误！')
        else:
            self.sim_list = sim_list
            self.fre_list = fre_list
            self.len = len(self.sim_list) or len(self.fre_list)
            self.lamda = lamda
            self.alpha = alpha
            self.beta = beta

    def ranking(self):
        """排序打分

        Return:
        排序得分               -- 得分序列，list类型
        """
        score_list = []
        for i in range(self.len):
            score = (1-self.lamda) * (e ** (-self.sim_list[i]*self.alpha)) + \
                    self.lamda * (1 / (1 + e **(-self.fre_list[i]*self.beta)))
            score_list.append(score)
        rank_list = []
        for index,i in enumerate(score_list):
            rank_list.append((index,i))
        rank_list.sort(key=lambda x:x[1],reverse=True)
        log('warning','最大值：{}，最小值{}，差值{}'.format(max(score_list),min(score_list),max(score_list)-min(score_list)))
        log('warning', '排序打分{}'.format(rank_list))
        return rank_list



if __name__ == "__main__":
    pass