# -*- coding: utf-8 -*-
"""
similarity主要完成了以下工作：
1.接受传入的查询词
2.对传入的查询词自动构造特征
3.加载训练好的SVM分类器进行相关度判别
"""
__title__ = 'QGDT-cpu'
__author__ = 'Ex_treme'
__license__ = 'MIT'
__copyright__ = 'Copyright 2018, Ex_treme'

from jieba import cut
from sklearn.externals import joblib

from QGDT.tools import get_current_path, jaccard, log


class Terms2Search(object):
    """查询词-->搜索词"""

    def __init__(self, search_list, svm_path):
        """
        Keyword arguments:
        clf                 -- 分类器，svm
        x1                  -- x1，查询词1
        x2                  -- x2，查询词2
        x3                  -- x3,查询词3
        """
        self.x1, self.x2, self.x3 = '', '', ''
        self.clf = joblib.load(get_current_path(svm_path))
        for index, i in enumerate(search_list):
            if index == 0:
                self.x1 = i
            elif index == 1:
                self.x2 = i
            elif index == 2:
                self.x3 = i

    def feature(self, x1, x2):
        """ 特征构造

        Return:
        [查询词1长度，查询词2长度，查询词12相关度]    -- 特征序列，list类型
        """
        x1_len = x1.__len__()
        x2_len = x2.__len__()
        jac = jaccard(x1, x2)
        return [x1_len, x2_len, jac]

    def corretale(self, x1, x2):
        """ 相关度判断

         Return:
         判断结果               -- 1为不相关（保留），0为相关（舍弃），bool类型
         """
        res = int(self.clf.predict([self.feature(x1, x2)])[0])
        if res == 1:
            return True
        if res == 0:
            return False

    def question(self, x):
        """ 问句判定

         Return:
         判断结果               -- True是问句，False不是问句，bool类型
         """
        for i in cut(x):
            if i in ['什么', '是什么', '哪些', '有哪些', '如何', '吗', '?', '？', ';', '；', '：', '：', ',', '，', '。', '.', '!', '！']:
                return True
        return False

    def correlation_calcuulate(self):
        """ 输出搜索词

         Return:
         问句或搜索词               -- 问句，str类型 搜索词，list类型
         """
        # 判断是否为问句
        if self.question(self.x1):
            log('warning', '生成问句：{} {} {}-->{}'.format(self.x1, self.x2, self.x3, self.x1))
            return self.x1
        if self.question(self.x2):
            log('warning', '生成问句：{} {} {}-->{}'.format(self.x1, self.x2, self.x3, self.x2))
            return self.x2
        if self.question(self.x3):
            log('warning', '生成问句：{} {} {}-->{}'.format(self.x1, self.x2, self.x3, self.x3))
            return self.x3
        # 前置条件
        terms = []
        if self.x1 is not '' and type(self.x1) == str:
            terms.append(self.x1)
        if self.x2 is not '' and type(self.x2) == str:
            terms.append(self.x2)
        if self.x3 is not '' and type(self.x3) == str:
            terms.append(self.x3)
        if terms.__len__() == 0:
            raise Exception('请传入至少一个查询词')
        # 相关度判定
        if terms.__len__() == 1:
            log('warning', '相关度计算：{}-->{}'.format(self.x1, self.x1))
            return [self.x1]
        if terms.__len__() == 2:
            if self.corretale(self.x1, self.x2):
                log('warning', '相关度计算：{} {}-->{} {}'.format(self.x1, self.x2, self.x1, self.x2))
                return [self.x1, self.x2]
            else:
                log('warning', '相关度计算：{} {}-->{}'.format(self.x1, self.x2, self.x2))
                return [self.x2]
        if terms.__len__() == 3:
            if self.corretale(self.x1, self.x2) and self.corretale(self.x2, self.x3):
                log('warning', '相关度计算：{} {} {}-->{} {} {}'.format(self.x1, self.x2, self.x3, self.x1, self.x2, self.x3))
                return [self.x1, self.x2, self.x3]
            elif not self.corretale(self.x1, self.x2) and self.corretale(self.x2, self.x3):
                log('warning', '相关度计算：{} {} {}-->{} {}'.format(self.x1, self.x2, self.x3, self.x2, self.x3))
                return [self.x2, self.x3]
            elif self.corretale(self.x1, self.x2) and not self.corretale(self.x2, self.x3):
                log('warning', '相关度计算：{} {} {}-->{} {}'.format(self.x1, self.x2, self.x3, self.x1, self.x3))
                return [self.x1, self.x3]
            elif not self.corretale(self.x1, self.x2) and not self.corretale(self.x2, self.x3):
                log('warning', '相关度计算：{} {} {}-->{}'.format(self.x1, self.x2, self.x3, self.x3))
                return [self.x3]


if __name__ == "__main__":
    pass
