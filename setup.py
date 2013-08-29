#!/usr/bin/env python

from distutils.core import setup
from distutils.util import convert_path
from fnmatch import fnmatchcase
import os

def find_packages(where='.'):
    out = []
    stack=[(convert_path(where), '')]
    while stack:
        where, prefix = stack.pop(0)
        for name in os.listdir(where):
            fn = os.path.join(where,name)
            if ('.' not in name and os.path.isdir(fn) and
                os.path.isfile(os.path.join(fn, '__init__.py'))
               ):
                out.append(prefix+name)
                stack.append((fn, prefix+name+'.'))
    for pat in ['ez_setup', 'distribute_setup']:
        out = [item for item in out if not fnmatchcase(item, pat)]
    return out


setup(name='decaf',
      version='0.9',
      description='Deep convolutional neural network framework',
      author='Yangqing Jia',
      author_email='jiayq84@gmail.com',
      packages=find_packages(),
      package_data={'decaf': ['util/_data/*'],
                    'decaf.demos.jeffnet': ['static/*', 'templates/*'],
                    'decaf.demos.notebooks': ['*.ipynb']},
     )
