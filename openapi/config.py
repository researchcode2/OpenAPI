#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals
import torch


if torch.cuda.is_available():
    DEVICE = torch.device('cuda:{}'.format(1))
else:
    DEVICE = torch.device('cpu')
