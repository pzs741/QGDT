# 基于深度学习和模板的问句生成算法（CPU版本）
---
（Question Generation Algorithm Based on Depth Learning and Template,QGDT）
=======

## 算法功能简介
基于深度学习和模板的问句生成算法完成了：通过训练好的SVM分类器对查询词（一至三个，可以是关键词、短语、句子）去重（查询词冗余）生成最终检索词，检索词通过排序算法（Word2Vec词向量模型+RNNLM语言模型）得到与预定义的模板的匹配得分，根据得分最高的模板生成最终的问句。

## 算法库组成
+ templates --- 预定义检索词模板
+ data --- 源数据（机器学习、深度学习）
+ models --- 预训练的模型（SVM、Word2Vec、RNNLM）
+ QGDT --- 训练模块、相关度判定模块、相似度计算模块、频度计算模块、排序生成模块

## 全局配置
self.LOG_ENABLE = True  # 是否开启日志
self.LOG_LEVEL  = 'WARNING' #默认日志等级
self.SVM_PATH = get_current_path('models/svm') #SVM默认路径
self.W2V_PATH = get_current_path('models/w2v')#Word2Vec默认路径
self.RNN_PATH = get_current_path('models/rnn')#RNNLM默认路径
self.DICT_PATH = get_current_path('models/rnn_dict')#字典默认路径
self.MAX_SAMPLE = 5#最大抽样次数
self.RANDOM = True#模板随机抽样
self.LAMBDA = 0.2#融合因子
self.ALPHA = 0.3#相似度计算调节因子
self.BETA = 0.5#频度计算调节因子


## 算法库安装(更新pip安装方式)
**pip全自动**
> conda install pytorch-cpu torchvision-cpu -c pytorch

> pip install QGDT-cpu


**git半自动安装**
https://github.com/pzs741/QGDT/tree/cpu
> $ git clone https://github.com/pzs741/QGDT.git
> $ cd QGDT-branch(cpu)    
> $ pip install -i https://pypi.douban.com/simple -r requirements.txt    
> $ python setup.py install  


**注意：**推荐这种安装方法，因为算法库依赖比较多，且模型较多，算法库体积较大，如遇错误，请按照requirements.txt中的依赖逐个安装！

## 算法库使用
1. 训练模块(train_models.py)

```
该模块独立，可以作为模型训练器单独使用，TrainSVM、TrainW2V、TrainRNN都继承TrainModel这个类，初始化的参数都为：origin_path(原始数据集相对路径)、train_path（转换后的训练集保存路径）、model_path（模型保存路径）
注意：RNNLM初始化多一个dict_path（数据字典保存路径），以上参数若不传递，则默认使用默认路径。
eg:以训练SVM为例
from QGDT import TrainRNN,TrainSVM,TrainW2V
#初始化一个类
t = TrainSVM()
#将原数据转换为训练集
t.origin_to_train()
#开始训练（训练完毕后模型自动保存到默认路径）
t.train()
```

2. 相关度判定模块(correlation_calculate.py)

```
该模块对查询词进行判定，舍弃冗余（对生成问句累赘）查询词，输出最终检索词。
eg:
from QGDT import Terms2Search
#初始化一个实例
t = Terms2Search(['Anti-DDoS流量清洗',' 攻击事件能否及时通知？'],'models/svm')
#计算搜索序列的相关度
t.correlation_calcuulate()
```

3. 相似度计算模块(similarity_calculate.py)

```
该模块完成了排序算法中相似度计算部分，该模块依赖于W2V词向量模型，使用WDM算法计算查询词与预定义模板的匹配度。
eg:
from QGDT import Search2Similar
#初始化一个实例
s = Search2Similar(['Anti-DDoS流量清洗',' 查询周防护统计情况','功能介绍'],'models/w2v')
#计算搜索词序列与预定义模板集的相似度
s.similarity_calculate()
```

4. 频度计算模块(frequency_calculate.py)

```
该模块完成了排序算法中频度计算部分，该模块依赖于RNNLM（RNN+LSTM语言模型）,通过一句话中每个单词输入模型，输出层参数可以反映下一个词的频度，将下一个单词作为参数进入模型输入层，最终为问句打分。

eg:
from QGDT import Search2Frequency
#初始化一个实例，传入语言模型和词典
s = Search2Frequency(['Anti-DDoS流量清洗',' 查询周防护统计情况','功能介绍'],'models/rnn','models/rnn_dict')
#计算搜索词序列的频度
s.frequency_calculate()
```

5. 排序生成模块(question_generate.py)

```
该模块将相关度判定模块、相似度计算模块、频度计算模块融合在一起，通过精心设计的信息检索模型为生成问句排序打分，最后由得分最高的模板生成相应问句。
eg:
from QGDT import QG
#相似度序列
sim_list = [2.261679196747406, 3.075403727118124, 2.871388395024239, 2.2647689008194836]
#频度序列
fre_list = [0.3126591742038727, 0.0, -0.19479990005493164, -0.19479990005493164]
#初始化实例
q = QG(sim_list,fre_list,0.2,0.3,0.5)
#排序打分
q.ranking()
```

## 实例测试

```
# -*- coding: utf-8 -*-
"""
A simple example, have fun!
"""
__title__ = 'QGDT'
__author__ = 'Ex_treme'
__license__ = 'MIT'
__copyright__ = 'Copyright 2018, Ex_treme'

import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))

from QGDT import QGDT

if __name__ == "__main__":
    test = [
    'MapReduce服务 如何使用MRS？                         ',
    'Anti-DDoS流量清洗 查询指定EIP防护状态 功能介绍               ',
    'MapReduce服务 终止集群 功能介绍                        ',
    '多维交互分析服务 获取日志列表 功能介绍                         ',
    '多维交互分析服务 新建文件夹 功能介绍                          ',
    '统一身份认证服务 获取用户token 功能介绍                      ',
    '统一身份认证服务 获取联邦认证的unscopedtoken 功能介绍           ',
    '镜像服务 异步任务查询 功能介绍                             ',
    '镜像服务 查询镜像成员列表视图（OpenStack原生） 功能介绍            ',
    'Anti-DDoS流量清洗 Token认证 调用接口步骤                 ',
    'Anti-DDoS流量清洗 查询指定EIP异常事件 响应                 ',
    'MapReduce服务 新增作业并执行 URI                      ',
    'MapReduce服务 解析响应消息 响应消息头                     ',
    'MapReduce服务 扩容集群 操作步骤                        ',
    'MapReduce服务 产品术语 ACL                         ',
    'MapReduce服务 产品术语 AM                          ',
    '多维交互分析服务 获取日志列表 URL                          ',
    '多维交互分析服务 获取日志列表 请求                           ',
    '多维交互分析服务 新建文件夹 响应码                           ',
    '多维交互分析服务 接口定义 日期与时间规范                        ',
    '多维交互分析服务 获取集群主机列表 响应码                        ',
    '统一身份认证服务 获取用户token URI                       ',
    '统一身份认证服务 获取用户token 请求                        ',
    '统一身份认证服务 查询权限的详细信息 状态码                       ',
    '统一身份认证服务 查询租户中用户组的权限 请求                      ',
    '统一身份认证服务 查询租户中用户组的权限 状态码                     ',
    '镜像服务 批量删除镜像成员 请求                             ',
    '镜像服务 批量删除镜像成员 响应                             ',
    '镜像服务 批量更新镜像成员状态 响应                           ',
    '镜像服务 异步任务查询 URI                              ',
    '镜像服务 查询镜像成员列表视图（OpenStack原生） URI             ',
    'Anti-DDoS流量清洗 Token认证 应用场景                   ',
    '多维交互分析服务 接口定义 请求方法                           ',
    '多维交互分析服务 API规范定义 接口适用范围                      ',
    '镜像服务 更新镜像信息 响应                               ',
    '镜像服务 更新镜像信息 返回值                              ',
    '镜像服务 查询镜像成员列表视图（OpenStack原生） 请求              ',
    'Anti-DDoS流量清洗 查询周防护统计情况                      ',
    'Anti-DDoS流量清洗 Token认证                        ',
    'Anti-DDoS流量清洗 查询指定EIP防护状态                    ',
    'Anti-DDoS流量清洗 附录                             ',
    'Anti-DDoS流量清洗 生成AK、SK                        ',
    '多维交互分析服务 获取日志列表                              ',
    '多维交互分析服务 新建文件夹                               ',
    '多维交互分析服务 导入导出数据                              ',
    '统一身份认证服务 获取用户token                           ',
    '统一身份认证服务 查询终端节点详情                            ',
    '统一身份认证服务 获取联邦认证的unscopedtoken                ',
    '统一身份认证服务 删除用户组中用户                            ',
    '镜像服务 生成AK、SK                                 ',
    '镜像服务 获取镜像成员详情                                ',
    '镜像服务 镜像复制                                    ',
    '镜像服务 AK/SK认证                                 ',
    '镜像服务 异步任务查询                                  ',
    '镜像服务 查询镜像成员列表视图（OpenStack原生）                 ',
    '多维交互分析服务                                     ',
    '镜像服务 镜像视图                                    ',
    'Anti-DDoS流量清洗 告警提醒API接口                      ',
    'Anti-DDoS流量清洗 开通Anti-DDoS服务 功能介绍             ',
    'Anti-DDoS流量清洗 查询Anti-DDoS配置可选范围 功能介绍         ',
    'Anti-DDoS流量清洗 查询Anti-DDoS服务 功能介绍             ',
    'MapReduce服务 MRS支持哪些作业类型？                     ',
    'MapReduce服务 Spark集群能访问OBS中的数据吗？              ',
    'MapReduce服务 MRS当前支持哪些规格主机？                   ',
    '多维交互分析服务 M-OLAP与Spark什么关系？                   ',
    '多维交互分析服务 M-OLAP简介                            ',
    'Anti-DDoS流量清洗 示例代码                           ',
    'Anti-DDoS流量清洗 资料下载                           ',
    'Anti-DDoS流量清洗 通用请求返回值                        ',
    'Anti-DDoS流量清洗 开通Anti-DDoS服务 URI              ',
    'Anti-DDoS流量清洗 开通Anti-DDoS服务 请求               ',
    'Anti-DDoS流量清洗 查询Anti-DDoS任务 请求               ',
    'Anti-DDoS流量清洗 查询周防护统计情况 响应                   ',
    'Anti-DDoS流量清洗 查询周防护统计情况 返回值                  ',
    'Anti-DDoS流量清洗 查询Anti-DDoS配置可选范围 URI          ',
    'Anti-DDoS流量清洗 公共消息头                          ',
    'Anti-DDoS流量清洗 查询Anti-DDoS服务 URI              ',
    'MapReduce服务 修订记录                             ',
    'MapReduce服务 终止集群                             ',
    'MapReduce服务 发起请求                             ',
    'MapReduce服务 查询作业exe对象列表                      ',
    'MapReduce服务 首次购买集群                           ',
    '多维交互分析服务 接口定义                                ',
    '多维交互分析服务 API规范定义                             ',
    '多维交互分析服务 M-OLAP简介 M-OLAP结构                   ',
    '统一身份认证服务 公共响应消息头                             ',
    '镜像服务 公共消息头                                   ',
    '镜像服务 请求签名流程 签名过程                             ',
    '镜像服务 公共请求消息头                                 ',
    '镜像服务 服务使用方法                                  ',
    '镜像服务 请求认证方式                                  ',
    'Anti-DDoS流量清洗 接口调用方法                         ',
    '多维交互分析服务 M-OLAP简介 M-OLAP特性                   ',
    '镜像服务 镜像视图 视图属性                               ',
    '统一身份认证服务 权限                                  ',
    '统一身份认证服务 版本信息                                ',
    'Anti-DDoS流量清洗 查询Anti-DDoS服务                  ',
    'MapReduce服务 购买MRS集群             ',
    'MapReduce服务 扩容集群                ',
    'Anti-DDoS流量清洗 查询Anti-DDoS配置可选范围 ',
    ]
    res_list = []
    for i in test:
        q = QGDT(i,lamda=0.2,alpha=0.3,beta=0.5)
        q.ranking_algorithm()
        res = q.question_generation()
        res_list.append(res)
    for i in res_list:
        print(i)
```


## 算法改进
+ 采用WDM距离算法计算相似度，检索词不局限于关键词，可以为短语，句子。
+ 相对于传统语言模型，深度学习语言模型为句子打分更具优势。
+ 新提出的问句生成模型除了传统模型的融合模型参数，增添了两个新的调节参数用于更好的调整模型，提高算法准确率。  
+ 首次将该模型用于知识库构建（web文档问答对自动生成）

## 参考文献
>  Zhao S, Wang H, Li C, et al. Automatically Generating Questions from Queries for Community-based Question Answering[C]// 2011:929--937.  
>  H Gao,C Guo∗,el al.Supervised Word Mover's Distance[C]//29th Conference on Neural Information Processing Systems (NIPS 2016), Barcelona, Spain.


---
## 作者
Z.S. Peng/[**Ex_treme**](https://pzs741.github.io/)


