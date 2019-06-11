#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from setuptools import setup, find_packages

setup(
    name = "openapi",
    version = "2.0",
    description = "Interpret Piecewise Linear Model Behind API",
    author = "Zicun Cong",
    author_email = "zcong@sfu.ca",
    zip_safe=False,
    packages=find_packages("."),
    include_package_data=True,
    package_data={'': ['*.json', '*.txt']},
    install_requires=["scipy", "numpy", "sklearn", "torchvision", "torch"]
)