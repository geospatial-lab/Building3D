import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


class APCalculator(object):
    def __init__(self, distance_thresh=0.1, confidence_thresh=0.7):
        r"""
        :param distance_thresh: the distance thresh
        :param confidence_thresh: the edges confident thresh
        """
        self.distance_thresh = distance_thresh
        self.confidence_thresh = confidence_thresh
        self.ap_dict = {'tp_corners': 0, 'num_pred_corners': 0, 'num_label_corners': 0, 'distance': 0, 'tp_edges': 0,
                        'num_pred_edges': 0, 'num_label_edges': 0, 'average_corner_offset': 0, 'corners_precision': 0,
                        'corners_recall': 0, 'corner_f1': 0, 'edges_precision': 0, 'edges_recall': 0, 'edges_f1': 0}

    def compute_metrics(self, batch):
        r"""
        :param batch: batch_size, predicted_corners, predicted_edges, predicted_score, wf_vertices, wf_edges, centroid,
        max_distance
        : test case
            batch_size = np.array([1])
            predicted_corners = np.array([[1, 2, 3], [7, 8, 9], [4, 5, 1], [7, 8, 9], [5,3,2], [1, 2, 4], [2, 5, 7]])
            label_corners = np.array([[2, 3, 4], [5, 6, 4], [6, 7, 8]])
            predicted_edges = np.array([[1, 2], [1, 5], [2, 4]])
            label_edges = np.array([[0, 1], [0, 2], [1, 2], [3, 4], [4, 5],[2, 4], [1, 3], [-1, -1], [-1, -1]])
            centroid = np.array([1, 2, 3])
            max_distance = np.array([[2]])
            predicted_score = np.array([[0.8, 0.8, 0.2, 1]]
        :return: AP Dict
        """
        batch_size = batch['batch_size']
        predicted_corners, predicted_edges, predicted_score = batch['predicted_corners'], batch['predicted_edges'], \
            batch['predicted_score']
        label_corners, label_edges = batch['wf_vertices'], batch['wf_edges']
        centroid, max_distance = batch['centroid'], batch['max_distance']

        for b in range(batch_size):
            # ----------------------- Confidence Thresh ---------------------------
            p_edges = predicted_edges[b][predicted_score[b] > self.confidence_thresh]
            predicted_edges_indices = p_edges.flatten()
            used_predicted_edges_indices = np.unique(predicted_edges_indices)
            p_corners = predicted_corners[b][used_predicted_edges_indices]
            index_mapping = {old_index: new_index for new_index, old_index in enumerate(used_predicted_edges_indices)}
            p_edges = np.vectorize(index_mapping.get)(p_edges)

            l_edges = label_edges[b][np.sum(label_edges[b], -1, keepdims=False) > 0]
            l_corners = label_corners[b][np.sum(label_corners[b], -1, keepdims=False) > -10]

            # ----------------------- Corners Eval based on Hungarian Mather algorithms ---------------------------
            # Calculate the distance matrix
            distance_matrix = cdist(p_corners, l_corners)
            predict_indices, label_indices = linear_sum_assignment(distance_matrix)
            mask = distance_matrix[predict_indices, label_indices] <= self.distance_thresh
            tp_corners_predict_indices, tp_corners_label_indices = predict_indices[mask], label_indices[mask]
            tp_corners = len(tp_corners_predict_indices)
            tp_fp_corners = len(p_corners)
            tp_fn_corners = len(l_corners)

            # ----------------------- Average Corners Offset---------------------------
            distances = np.linalg.norm((p_corners[tp_corners_predict_indices] * max_distance[b] + centroid[b]) - (
                    l_corners[tp_corners_label_indices] * max_distance[b] + centroid[b]), axis=1)
            distances = np.sum(distances)

            # ------------------------------- Edges Eval ------------------------------
            corners_map = {key: value for key, value in
                           zip(tp_corners_predict_indices[mask], tp_corners_label_indices[mask])}
            for i, _ in enumerate(p_edges):
                for j in range(2):
                    p_edges[i, j] = corners_map[p_edges[i, j]] if p_edges[i, j] in corners_map else -1
                p_edges[i] = sorted(p_edges[i])

            tp_edges = np.sum([np.any(np.all(e == l_edges, axis=1)) for e in predicted_edges[b]])
            tp_fp_edges = len(p_edges)
            tp_fn_edges = len(l_edges)
            # precision = tp_edges / tp_fp
            # recall = tp_edges / tp_fn

            # ------------------------------- Return AP Dict ------------------------------
            self.ap_dict['tp_corners'] += tp_corners
            self.ap_dict['num_pred_corners'] += tp_fp_corners
            self.ap_dict['num_label_corners'] += tp_fn_corners

            self.ap_dict['distance'] += distances

            self.ap_dict['tp_edges'] += tp_edges
            self.ap_dict['num_pred_edges'] += tp_fp_edges
            self.ap_dict['num_label_edges'] += tp_fn_edges

    def output_accuracy(self):
        self.ap_dict['average_corner_offset'] = self.ap_dict['distance'] / self.ap_dict['tp_corners']
        self.ap_dict['corners_precision'] = self.ap_dict['tp_corners'] / self.ap_dict['num_pred_corners']
        self.ap_dict['corners_recall'] = self.ap_dict['tp_corners'] / self.ap_dict['num_label_corners']
        self.ap_dict['corners_f1'] = 2 * self.ap_dict['corners_precision'] * self.ap_dict['corners_recall'] / (
                self.ap_dict['corners_precision'] + self.ap_dict['corners_recall'])

        self.ap_dict['edges_precision'] = self.ap_dict['tp_edges'] / self.ap_dict['num_pred_edges']
        self.ap_dict['edges_recall'] = self.ap_dict['tp_edges'] / self.ap_dict['num_label_edges']
        self.ap_dict['edges_f1'] = 2 * self.ap_dict['edges_precision'] * self.ap_dict['edges_recall'] / (
                    self.ap_dict['edges_precision'] + self.ap_dict['edges_recall'])

        print('Average Corner offset', self.ap_dict['average_corner_offset'])
        print('Corners Precision: ', self.ap_dict['corners_precision'])
        print('Corners Recall: ', self.ap_dict['corners_recall'])
        print('Corners F1：', self.ap_dict['corners_f1'])

        print('Edges Precision: ', self.ap_dict['edges_precision'])
        print('Edges Recall: ', self.ap_dict['edges_recall'])
        print('Edges F1: ', self.ap_dict['edges_f1'])

    def reset(self):
        self.ap_dict = {'tp_corners': 0, 'num_pred_corners': 0, 'num_label_corners': 0, 'distance': 0, 'tp_edges': 0,
                        'num_pred_edges': 0, 'num_label_edges': 0, 'average_corner_offset': 0, 'corners_precision': 0,
                        'corners_recall': 0, 'corners_f1': 0, 'edges_precision': 0, 'edges_recall': 0, 'edges_f1': 0}
