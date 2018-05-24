#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  setup.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 05.12.2018
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
  name = 'Hello world app',
  ext_modules = cythonize("cvb0_main.pyx"),
  include_dirs = [numpy.get_include()]
)

