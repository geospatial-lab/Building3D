# Building3D

## 1. Dataset Description
#### Point Clouds file
```
# x, y, z, r, g, b, nir, intensity
537345.7100 6583739.3900 32.5400 59 63 66 58 0.4900
537345.4800 6583739.9600 32.4700 52 55 60 58 0.5100
537345.4000 6583739.6500 32.5100 50 54 57 48 0.7600
537345.1700 6583739.7200 32.5100 50 53 58 52 0.2300
537344.9500 6583739.8500 32.5300 51 55 58 50 0.9800
....
```
#### Wireframe file
```
# v: vertex   l: line or edge
v 535251.8199996186 6580870.999999885 44.42389989852905
v 535252.1199998093 6580879.400000458 44.40930009841919
v 535243.3900000007 6580871.320000057 43.34209991455078
....
l 1 2
l 1 3
....
```
## 2. Eval
```python
class APCalculator(object):
    def __init__(self, distance_thresh=0.1, confidence_thresh=0.7):
        r"""
        :param distance_thresh: the distance thresh
        :param confidence_thresh: the edges confident thresh
        """
        self.distance_thresh = distance_thresh
        self.confidence_thresh = confidence_thresh
    
    def compute_metrics(self, batch):
        # ....
        return None
    
    def output_accuracy(self):
        # ....
        return None
    
    def reset(self):
        # ....
        return None
        

# --------------------------------- Input Case -------------------------------------
batch = {'batch_size': 1,
         'predicted_corners': np.array([[[1, 2, 3], [7, 8, 9], [4, 5, 1], [7, 8, 9], [5,3,2], [1, 2, 4], [2, 5, 7], [1, 1, 1]]]),
         'wf_vertices': np.array([[[2, 3, 4], [5, 6, 4], [6, 7, 8],[-10, -10, -10], [-10, -10, -10]]]),
         'predicted_edges': np.array([[[1, 2], [1, 5], [5, 6], [2, 4]]]),
         'wf_edges': np.array([[[0, 1], [0, 2], [1, 2], [3, 4], [4, 5],[2, 4], [1, 3], [-1, -1], [-1, -1]]]),
         'predicted_score': np.array([[0.8, 0.8, 0.2, 1]]),
          'centroid': np.array([[2, 2, 2]]),
         'max_distance': np.array([[1]])}

# --------------------------------- Output -------------------------------------
# Average Corner offset (ACO)
# Corners Precision (CP), Corners Recall (CR), Corners F1 (C_F1)
# Edges Precision (EP), Edges Recall (ER), Edges F1 (E_F1)
print('Average Corner offset', self.ap_dict['average_corner_offset'])
print('Corners Precision: ', self.ap_dict['corners_precision'])
print('Corners Recall: ', self.ap_dict['corners_recall'])
print('Corners F1：', self.ap_dict['corners_f1'])

print('Edges Precision: ', self.ap_dict['edges_precision'])
print('Edges Recall: ', self.ap_dict['edges_recall'])
print('Edges F1: ', self.ap_dict['edges_f1'])
```