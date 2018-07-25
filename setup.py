#!/bin/python3.6
# -*- coding: utf-8 -*-
"""
Ex_treme 2018 -- https://github.com/pzs741
"""
__title__ = 'QGDT-cpu'
__author__ = 'Ex_treme'
__license__ = 'MIT'
__copyright__ = 'Copyright 2018, Ex_treme'


version_info = (0, 3, 4)

__version__ = ".".join(map(str, version_info))


import sys
import os
import codecs

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup



with open('requirements.txt') as f:
    required = f.read().splitlines()


if sys.argv[-1] == 'publish':
    os.system('python3 setup.py sdist upload -r pypi')
    sys.exit()


# This *must* run early. Please see this API limitation on our users:
# https://github.com/codelucas/newspaper/issues/155
if sys.version_info[0] == 2 and sys.argv[-1] not in ['publish', 'upload']:
    sys.exit('WARNING! You are attempting to install QGDT-cpu\'s '
             'python3 repository on python2. PLEASE RUN '
             '`$ pip3 install QGDT-cpu` for python3 or '
             '`$ pip install QGDT-cpu` for python2')



with codecs.open('README.md', 'r', 'utf-8') as f:
    readme = f.read()


setup(
    name=__title__,
    author=__author__,
    license=__license__,
    version=__version__,
    long_description=readme,
    include_package_data=True,
    install_requires=required,
    description='Question Generation Algorithm Based on Depth Learning and Template,QGDT',
    author_email='pzsyjsgldd@163.com',
    url='https://github.com/pzs741/QGDT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Natural Language :: English',
        'Intended Audience :: Developers',
    ],
    packages = ['QGDT'],
    package_dir={'QGDT':'QGDT'},
    package_data={'QGDT':['*.*','data/*','models/*','templates_x/*']}

)
