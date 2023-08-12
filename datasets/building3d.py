#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023-08-11 3:06 p.m.
# @Author  : shangfeng
# @Organization: University of Calgary
# @File    : building3d.py.py
# @IDE     : PyCharm

import os
import sys
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class Building3DReconstructionDataset(Dataset):
    def __init__(self, dataset_config, split_set, logger=None):
        self.dataset_config = dataset_config
        self.roof_dir = dataset_config.root_dir
        self.num_points = dataset_config.num_points
        self.use_color = dataset_config.use_color
        self.use_intensity = dataset_config.use_intensity
        self.normalize = dataset_config.normalize
        self.augment = dataset_config.augment

        assert split_set in ["train", "test"]
        self.split_set = split_set

        self.pc_files, self.wireframe_files = self.load_files()

        if logger:
            logger.info("Total Sample: %d" % len(self.pc_files))

    def __len__(self):
        return len(self.pc_files)

    def __getitem__(self, index):
        pc_file = self.pc_files[index]
        wireframe_file = self.wireframe_files[index]

        # load point clouds & wireframe
        pc = np.loadtxt(pc_file, dtype=np.float64)
        wf_vertices, wf_edges = self.load_wireframe(wireframe_file)

        # point clouds processing
        if not self.use_color:
            point_cloud = pc[:, 0:3]
        elif self.use_color and not self.use_intensity:
            point_cloud = pc[:, 0:7]
            point_cloud[:, 3:] = point_cloud[:, 3:] / 256.0
        elif not self.use_color and self.use_intensity:
            point_cloud = np.concatenate((pc[:, 0:3], pc[:, 7]), axis=1)
        else:
            point_cloud = pc
            point_cloud[:, 3:7] = point_cloud[:, 3:7] / 256.0

        print(pc_file)
        print(point_cloud[0])
        #
        # if self.transform:
        #     x = self.transform(x)
        return 0

    def load_files(self):
        data_dir = os.path.join(self.roof_dir, self.split_set)
        pc_files = [pc_file for pc_file in glob.glob(os.path.join(data_dir, 'xyz', '*.xyz'))]
        wireframe_files = [wireframe_file for wireframe_file in glob.glob(os.path.join(data_dir, 'wireframe', '*.obj'))]
        return pc_files, wireframe_files

    def load_wireframe(self, wireframe_file):
        vertices = []
        edges = set()
        with open(wireframe_file) as f:
            for lines in f.readlines():
                line = lines.strip().split(' ')
                if line[0] == 'v':
                    vertices.append(line[1:])
                else:
                    obj_data = np.array(line[1:], dtype=np.int32).reshape(2) - 1
                    edges.add(tuple(sorted(obj_data)))
        vertices = np.array(vertices, dtype=np.float64)
        edges = np.array(list(edges))
        return vertices, edges

    def print_self_values(self):
        attributes = vars(self)
        for attribute, value in attributes.items():
            print(attribute, "=", value)
