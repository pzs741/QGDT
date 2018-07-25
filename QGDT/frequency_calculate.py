# -*- coding: utf-8 -*-
"""
frequency_calculate 主要完成了：
1.加载RNNLM模型
2.为搜索词匹配模板
3.为生成的问句打分
"""
__title__ = 'QGDT-cpu'
__author__ = 'Ex_treme'
__license__ = 'MIT'
__copyright__ = 'Copyright 2018, Ex_treme'

import pickle
import torch
from jieba import cut
from QGDT.templates_init import x0_1,x0_2,x0_3
from QGDT.tools import log,num_layers, hidden_size, device,get_current_path

#搜索词匹配模板
def searchs2templates(s1='',s2='',s3='',rank_id = None):
    """通过搜索词，匹配相应搜索词个数的模板

    Keyword arguments:
    s1                  -- str类型，搜索词1
    s2                  -- str类型，搜索词2
    s3                  -- str类型,搜索词3
    rank_id             -- 未定义，排序打分后得到的模板id号
    Return:
        [问句1,问句2,问句3,问句4]
    """
    if s1 and s2 and s3:
        t =3
        t1 = s1+s2+x0_3[0].split()[0]+s3+x0_3[0].split()[1]
        t2 = s1+s2+x0_3[1].split()[0]+s3+x0_3[1].split()[1]
        t3 = s1+s2+x0_3[2].split()[0]+s3+x0_3[2].split()[1]
        t4 = s1+s2+x0_3[3].split()[0]+s3+x0_3[3].split()[1]
    elif s1 and s2:
        t = 2
        t1 = s1 + x0_2[0].split()[0] + s2 + x0_2[0].split()[1]
        t2 = s1 + x0_2[1].split()[0] + s2 + x0_2[1].split()[1]
        t3 = s1 + x0_2[2].split()[0] + s2 + x0_2[2].split()[1]
        t4 = s1 + x0_2[3].split()[0] + s2 + x0_2[3].split()[1]
    elif s1:
        t = 1
        t1 = s1 + x0_1[0]
        t2 = x0_1[1].split()[0]+ s1 + x0_1[1].split()[1]
        t3 = s1 + x0_1[2]
        t4 = s1 + x0_1[3]
    else:
        raise Exception('搜索词提取错误！')
    t_list = [t1,t2,t3,t4]
    if  rank_id == None:
        return t_list
    else:
        log('warning','{} {} {} --> {}'.format(s1,s2,s3,t_list[rank_id]))
        return t_list[rank_id]



class Search2Frequency(object):
    """搜索词-->生成频率"""
    def __init__(self,search_list,*args):
        """
       Keyword arguments:
       rnnlm                    -- 语言模型，RNN+LSTM
       question                 -- 候选生成问句
       """

        self.rnnlm = torch.load(get_current_path(args[0]))
        with open(get_current_path(args[1]), mode='rb') as f:
            self.dict = pickle.load(f)
            self.s1, self.s2, self.s3 = '', '', ''
            for index, i in enumerate(search_list):
                if index == 0:
                    self.s1 = i
                elif index == 1:
                    self.s2 = i
                elif index == 2:
                    self.s3 = i
            self.questions = searchs2templates(s1=self.s1,s2=self.s2,s3=self.s3)


    def question2probability(self,t):
        """  单个问句生成概率

        Return:
        probability      --生成问句得分，float格式
        """

        question = [i.lower() for i in cut(t) ]
        with torch.no_grad():
            #初始化神经网络和输入概率
            state = (torch.zeros(num_layers, 1, hidden_size).to(device),
                     torch.zeros(num_layers, 1, hidden_size).to(device))
            prob = torch.zeros(len(self.dict))
            prob[self.dict['<eos>']] = 1.0
            input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)

        #计算每个词的概率得分
        score_list = []
        for index,i in enumerate(question):
            output, state = self.rnnlm(input, state)
            try:
                word_id = self.dict[i]
            except:
                score_list.append(0.0)
                continue
            score_list.append(float(output[0][word_id]))
            input.fill_(word_id)
        return sum(score_list)

    def frequency_calculate(self):
        """  计算所有模板生成问句的生成频度

        Return:
        frequency      --生成频度，list格式
        """
        fq_list = []
        if self.questions.__len__() is not 4:
            raise Exception('模板数不为4,请重新配置！')
        else:
            for i in range(4):
                fq_list.append(self.question2probability(self.questions[i]))

        log('warning','频度计算：{}-->{}'.format(self.questions,fq_list))

        return fq_list

if __name__ == "__main__":
    pass