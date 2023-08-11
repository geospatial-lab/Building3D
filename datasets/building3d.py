#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023-08-11 3:06 p.m.
# @Author  : shangfeng
# @Organization: University of Calgary
# @File    : building3d.py.py
# @IDE     : PyCharm

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset


class Building3DReconstructionDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)

        return x, y