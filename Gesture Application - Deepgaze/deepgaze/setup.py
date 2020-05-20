#!/usr/bin/env python

from distutils.core import setup

setup(name='deepgaze',
  version='0.1',
  url='https://github.com/mpatacchiola/deepgaze',
  description='Head pose and Gaze estimation with Convolutional Neural Networks',
  author='Massimiliano Patacchiola',
  packages = ['deepgaze'],
  package_data={'deepgaze': ['Readme.md']},
  include_package_data=True,
  license="The MIT License (MIT)",
  requires = ['numpy', 'cv', 'cv2', 'tensorflow']
 )
