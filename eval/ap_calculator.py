import os
import numpy as np
import torch


class APCalculator(object):
    def __init__(self, confidence_thresh=0.7):
        r"""
        :param confidence_thresh: the edges confident thresh
        ap_dict: num_pred_edges == tp + fp (edges)
                 num_label_edges == tp + fn (edges)
        """
        self.confidence_thresh = confidence_thresh
        self.ap_dict = {'tp_pts': 0, 'num_label_pts': 0, 'num_pred_pts': 0, 'pts_bias': np.zeros(3, np.float),
                        'tp_edges': 0, 'num_label_edges': 0, 'num_pred_edges': 0}

    def compute_metrics(self, batch):
        r"""
        :return: An AP dict
        """

        # ----------------------- Corners Eval based on Hungarian Mather algorithms ---------------------------

        predicted_labels = [[0, 1], [1, 2], [2, 3]]
        true_labels = [[0, 1], [2, 3], [1, 4], [3, 4], [4, 5], [2, 4], [1, 3]]

        # ------------------------------- Edges Eval ------------------------------
        tp_edges = np.sum([e in true_labels for e in predicted_labels])
        tp_fp_edges = len(predicted_labels)
        tp_fn_edges = len(true_labels)
        # precision = tp_edges / tp_fp
        # recall = tp_edges / tp_fn

        #  ------------------------------- Return AP Dict ------------------------------
        self.ap_dict['tp_edges'] += tp_edges
        self.ap_dict['num_pred_edges'] += tp_fp_edges
        self.ap_dict['num_label_edges'] += tp_fn_edges

    def reset(self):
        self.ap_dict = {'tp_pts': 0, 'num_label_pts': 0, 'num_pred_pts': 0, 'pts_bias': np.zeros(3, np.float),
                        'tp_edges': 0, 'num_label_edges': 0, 'num_pred_edges': 0}
