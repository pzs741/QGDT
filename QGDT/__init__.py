# -*- coding: utf-8 -*-
"""
努力不需要理由，如果需要，就是为了不需要的理由。
"""

__title__ = 'QGDT-cpu'
__author__ = 'Ex_treme'
__license__ = 'MIT'
__copyright__ = 'Copyright 2018, Ex_treme'

import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))

from QGDT.correlation_calculate import Terms2Search
from QGDT.frequency_calculate import Search2Frequency
from QGDT.qgdt import QGDT
from QGDT.question_generate import QG
from QGDT.similarity_calculate import Search2Similar
from QGDT.tools import (chinese,
                        device,
                        embed_size,
                        hidden_size,
                        num_layers,
                        num_epochs,
                        batch_size,
                        seq_length,
                        learning_rate,
                        log,
                        get_current_path,
                        jaccard,
                        Dictionary,
                        Corpus,
                        RNNLM,
                        extend_config,
                        Configuration, )
from QGDT.train_models import TrainRNN, TrainSVM, TrainW2V
from QGDT.templates import t1,t2,t3,t4

version_info = (0, 3, 4)

__version__ = ".".join(map(str, version_info))

print('__title__:',__title__)
print('__author__:',__author__)
print('__license__:',__license__)
print('__copyright__:',__copyright__)
print('__version__:',__version__)

import jieba

topic_list = [
    '云审计使用攻略',
    '机入侵检测使用攻略',
    '密钥管理服务',
    '降低企业沟通成本',
    '弹性负载均衡热门问题集锦',
    '云硬盘备份服务使用攻略',
    '数据快递服务',
    '实时流计算服务',
    '工具',
    '主机漏洞检测',
    '关系型数据库',
    'CDN',
    '合规中心',
    '地区和终端节点',
    '对象存储服务',
    '消息与短信',
    '云监控服务',
    '数据调度服务',
    '对象存储服务热门问题集锦',
    'SDK',
    'OBS_CSDK',
    '专属云使用攻略',
    'OBS_JAVA_SDK',
    'Anti',
    '软件开发云',
    '弹性伸缩服务',
    '渗透测试',
    '对象存储服务快速入门',
    '人工智能服务',
    '新手指南',
    '分布式数据库中间件',
    '证书管理',
    '函数服务',
    '虚拟私有云',
    '镜像服务',
    '云目录服务',
    '弹性云服务器',
    '云监控使用攻略',
    '云解析服务',
    '虚拟私有云使用指南',
    '开发者工具中心',
    'OBS_Python_SDK',
    '备案简介',
    '云报表服务',
    '华为云服务',
    '会议',
    'Web应用防火墙使用攻略',
    '易用',
    '安全态势感知服务',
    '数据接入服务',
    '统一身份认证服务',
    '分布式缓存服务',
    '即时通信',
    '多维交互分析服务',
    '云容器引擎',
    'OBS_PHP_SDK',
    '商业合作伙伴最终用户须知',
    '网站备案中心',
    '产品价格详情',
    '裸金属服务器',
    '弹性云服务器热门问题集锦',
    '产品新特性',
    '消息通知服务',
    '程序运行认证服务',
    'Anti-DDoS流量清洗',
    '数据仓库服务',
    '消息中心',
    '分布式消息服务',
    '常见问题',
    '文档数据库服务',
    '安全指数服务',
    'Web漏洞扫描使用攻略',
    '弹性云服务器使用攻略',
    '弹性负载均衡使用攻略',
    '弹性负载均衡',
    'OBS_Ruby_SDK',
    '云审计服务',
    '政策法规',
    '弹性伸缩服务使用攻略',
    '云容灾',
    '云硬盘',
    '云硬盘备份',
    '弹性文件服务',
    '机器学习服务',
    '专属云',
    '通信平台云',
    '软件开发云使用攻略',
    'MapReduce服务',
    '联络中心',
    '安全指数使用攻略',
    '云服务器备份',
    '数据查询服务',
    '云硬盘使用指南',
    '微服务云应用平台',
    'OBS_Android_SDK',
    '新手指引',
    '云专线',
    '主机入侵检测',
    '综合上云迁移交付服务',
    '如何',
    '是什么',
    '有哪些',
]
for i in topic_list:
    jieba.add_word(i.lower())
    jieba.suggest_freq(i.lower(), True)
log('info','初始化用户词典完成！')




