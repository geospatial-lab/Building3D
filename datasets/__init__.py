#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023-08-11 2:41 p.m.
# @Author  : shangfeng
# @Organization: University of Calgary
# @File    : __init__.py.py
# @IDE     : PyCharm

from .building3d import Building3DReconstructionDataset


def build_dataset(dataset_config):
    datasets_dict = {
        "train": Building3DReconstructionDataset(dataset_config, split_set="train"),
        "test": Building3DReconstructionDataset(dataset_config, split_set="test"),
    }

    return datasets_dict
