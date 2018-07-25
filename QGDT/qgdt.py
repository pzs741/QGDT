# -*- coding: utf-8 -*-
"""
QGDT（Question Generation Algorithm Based on DepthLearning and Template）主要完成了：
1.模型加载         -- 分类模型：SVM 词向量模型模型：word2vec 语言模型:RNNLM
2.排序算法         -- 基于相似度（WMD）和频度（RNNLM）的排序算法
3.问句生成         -- 基于模板生成问句
"""

__title__ = 'QGDT-cpu'
__author__ = 'Ex_treme'
__license__ = 'MIT'
__copyright__ = 'Copyright 2018, Ex_treme'

import logging

from QGDT.correlation_calculate import Terms2Search
from QGDT.frequency_calculate import Search2Frequency, searchs2templates
from QGDT.question_generate import QG
from QGDT.similarity_calculate import Search2Similar
from QGDT.tools import Configuration, extend_config,variance

class QGDT(object):
    """ 基于深度学习和模板的问句生成算法 """

    def __init__(self, search, config=None, *args, **kwargs):
        """
        Keyword arguments:
        search                   -- 查询词，str类型
        config                  -- 默认配置，可拓展，dict类型
        """
        self._config = config or Configuration()
        self._config = extend_config(self._config, kwargs)
        if self._config.LOG_ENABLE:
            logging.basicConfig(format='%(levelname)s:%(message)s', level=self._config.LOG_LEVEL)
        self.search_list = search.split()
        self.svm_path = self._config.SVM_PATH
        self.w2v_path = self._config.W2V_PATH
        self.dict_path = self._config.DICT_PATH
        self.rnn_path = self._config.RNN_PATH
        self.lamda = self._config.LAMBDA
        self.alpha = self._config.ALPHA
        self.beta = self._config.BETA
        self.random = self._config.RANDOM_SAMPLE

        self.question = ''
        self.sim_list = []
        self.fre_list = []
        self.rank_list = []

        self.diff = 0.0
        self.confusion = 0.0

        for i in args:
            if i[0] == 'svm':
                self.svm_path = i[1]
            elif i[0] == 'w2v':
                self.w2v_path = i[1]
            elif i[0] == 'dict':
                self.dict_path = i[1]
            elif i[0] == 'rnn':
                self.rnn_path = i[1]

    def ranking_algorithm(self):
        """排序算法

        Return:
        排序得分               -- 得分序列，list类型
        """
        t2s = Terms2Search(self.search_list, self.svm_path)
        self.search_list = t2s.correlation_calcuulate()
        if type(self.search_list) == str:
            self.question = self.search_list
        elif self.lamda == 0:
            s2s = Search2Similar(self.search_list, self.w2v_path,self.random)
            self.sim_list = s2s.similarity_calculate()
            self.fre_list = [0.0 for i in range(len(self.sim_list))]
            q = QG(self.sim_list, self.fre_list, self.lamda, self.alpha, self.beta)
            self.rank_list = q.ranking()
        elif self.lamda == 1:
            s2f = Search2Frequency(self.search_list, self.rnn_path, self.dict_path)
            self.fre_list = s2f.frequency_calculate()
            self.sim_list = [0.0 for i in range(len(self.fre_list))]
            q = QG(self.sim_list, self.fre_list, self.lamda, self.alpha, self.beta)
            self.rank_list = q.ranking()
        else:
            s2s = Search2Similar(self.search_list, self.w2v_path,self.random)
            self.sim_list = s2s.similarity_calculate()
            s2f = Search2Frequency(self.search_list, self.rnn_path, self.dict_path)
            self.fre_list = s2f.frequency_calculate()
            q = QG(self.sim_list, self.fre_list, self.lamda, self.alpha, self.beta)
            self.rank_list = q.ranking()

        if self.rank_list:
            self.diff = self.rank_list[0][1] - self.rank_list[-1][1]
            self.confusion = self.diff/variance([i[1] for i in self.rank_list])

    def question_generation(self):
        """问句生成

        Return:
        问句               -- str类型
        """
        if self.question:
            return self.question
        elif self.rank_list:
            s1, s2, s3 = '', '', ''
            for index, i in enumerate(self.search_list):
                if index == 0:
                    s1 = i
                elif index == 1:
                    s2 = i
                elif index == 2:
                    s3 = i
            res = searchs2templates(s1, s2, s3, self.rank_list[0][0])
            return res


if __name__ == "__main__":
    q = QGDT('Anti-DDoS流量清洗 查询Anti-DDoS配置可选范围 ', )
    q.ranking_algorithm()
    q.question_generation()
