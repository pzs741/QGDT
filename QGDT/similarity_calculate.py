# -*- coding: utf-8 -*-
"""
similarity_calculate 主要完成了：
1.加载word2vec模型
2.WMD计算短语相似度
3.相似度-->模板匹配度
"""
import random

__title__ = 'QGDT-cpu'
__author__ = 'Ex_treme'
__license__ = 'MIT'
__copyright__ = 'Copyright 2018, Ex_treme'


from gensim.models import Word2Vec
from TEDT.segmentation import WordSegmentation

from QGDT.tools import get_current_path, log, Configuration
from QGDT.templates import t1,t2,t3,t4

t1 = set(t1)
t2 = set(t2)
t3 = set(t3)
t4 = set(t4)

t3.difference_update(t2)

t4.difference_update(t2)
t4.difference_update(t3)

t1.difference_update(t2)
t1.difference_update(t3)
t1.difference_update(t4)

log('info','模板集初始化完成！')


# 匹配模板
def templates(x1='', x2='', x3=''):
    """接受得到的搜索词，匹配相应搜索词个数的模板

    Keyword arguments:
    x1                  -- x1，搜索词1
    x2                  -- x2，搜索词2
    x3                  -- x3,搜索词3
    Return:
        (搜索序列，模板号)
    """
    if x1 and x2 and x3:
        s1 = x1 + ' ' + x2 + ' ' + x3
        return s1, 3
    elif x1 and x2:
        s1 = x1 + ' ' + x2
        return s1, 2
    elif x1:
        s1 = x1
        return s1, 1
    else:
        raise Exception('未正确输入查询词')


class Search2Similar(object):
    """搜索词-->生成相似度"""

    def __init__(self, search_list, w2v_path,random):
        """
        Keyword arguments:
        w2v                 -- 词嵌入，word2vec
        templates           -- 搜索序列和模板号
        """
        self.x1, self.x2, self.x3 = '', '', ''
        self.w2v = Word2Vec.load(get_current_path(w2v_path))
        for index, i in enumerate(search_list):
            if index == 0:
                self.x1 = i
            elif index == 1:
                self.x2 = i
            elif index == 2:
                self.x3 = i
        self.templates = templates(x1=self.x1, x2=self.x2, x3=self.x3)
        self.random = random

    def search2keywords(self, s):
        """ 查询词转换成关键词

        Return:
        k      --关键词序列，list格式
        """
        w = WordSegmentation()
        k = w.segment(s)
        return k

    def wmd(self, s1, s2):
        """ 计算两个搜索词之间的WMD距离
        注意：当词嵌入中不存在计算词时，返回'inf',意为距离无穷大。
        Return:
        wmd      --wmd距离（wmd越小，表示搜索词越相似），float格式
        """
        k1 = self.search2keywords(s1)
        k2 = self.search2keywords(s2)
        wmd = self.w2v.wmdistance(k1,k2)
        return wmd

    def mul_wmd(self, s1, s2):
        """ 计算多个查询词的平均WMD距离

        Return:
        avg_wmd      --平均wmd距离（wmd越小，表示搜索词越相似），float格式
        """
        if s1.split().__len__() is not s2.split().__len__():
            return float('inf')
        else:
            s1 = s1.split()
            s2 = s2.split()
            count = len(s1 or s2)

        mul_wmd = []
        for i in range(count):
            mul_wmd.append(self.wmd(s1[i], s2[i]))
        return sum(mul_wmd) / mul_wmd.__len__()

    def tmp_shuffle(self,tmp_list):
        """ 模板随机抽样

        Return:
        tmp_list         --随机排列的模板，list格式
        """
        tmp_list = list(tmp_list)

        len1 = len(tmp_list)
        if self.random:
            random_length = random.randint(1, len(tmp_list))
            tmp_list = tmp_list[:random_length]
            random.shuffle(tmp_list)
        len2 = len(tmp_list)
        log('info','少计算{}个模板！'.format(len1-len2))
        return tmp_list

    def similarity_calculate(self):
        """ 搜索序列与模板之间的相似度计算

        Return:
        templates_id        --对应搜索词个数的模板，int格式
        min_wmd             --最小wmd距离（wmd越小，表示搜索词越相似），float格式
        template_id         --最后使用模板id，int格式
        """
        wmd_list = []
        min_wmd_list = []
        s1 = self.templates[0]
        for s2 in self.tmp_shuffle(t1):
            wmd_list.append(self.wmd(s1.split(' ')[-1], s2))
        min_wmd_list.append(min(wmd_list))
        wmd_list = []
        for s2 in self.tmp_shuffle(t2):
            wmd_list.append(self.wmd(s1.split(' ')[-1], s2))
        min_wmd_list.append(min(wmd_list))
        wmd_list = []
        for s2 in self.tmp_shuffle(t3):
            wmd_list.append(self.wmd(s1.split(' ')[-1], s2))
        min_wmd_list.append(min(wmd_list))
        wmd_list = []
        for s2 in self.tmp_shuffle(t4):
            wmd_list.append(self.wmd(s1.split(' ')[-1], s2))
        min_wmd_list.append(min(wmd_list))
        log('warning', '相似度计算：{}-->{}'.format(s1, min_wmd_list))

        return min_wmd_list


if __name__ == "__main__":
    pass
