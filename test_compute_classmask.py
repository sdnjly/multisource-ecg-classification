import numpy as np

import category_mapping
from data_prepare import compute_class_mask

labels = np.array([[0, 0, 0, 1],
                   [0, 0, 1, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]])

mapping_mat = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 1, 1, 0],
                        [0, 0, 1, 1]])

mapping_mat_fixed = category_mapping.complete_mapping_matrix(mapping_mat, 4)
print(mapping_mat_fixed)

labels_offline_mapped = np.matmul(labels, mapping_mat_fixed)
print(labels_offline_mapped)

class_mask = compute_class_mask(mapping_mat_fixed, labels_offline_mapped, True)
print(class_mask)
