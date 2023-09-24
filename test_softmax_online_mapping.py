import numpy as np

from scipy.special import softmax

import category_mapping
from data_prepare import compute_class_mask


def sigmoid(X):
   return 1/(1+np.exp(-X))


def relu(X):
   return np.maximum(0,X)


def online_mapping(prediction, mapping_mat, categories):
    # branch 1: mapping by summary
    prediction_sigmoid = sigmoid(prediction)
    prediction_b1 = np.matmul(prediction_sigmoid, mapping_mat)
    prediction_b1 = np.clip(prediction_b1, 0, 1)

    # branch 2: mapping by softmax and maximum
    prediction = np.expand_dims(prediction, axis=2)
    prediction = np.tile(prediction, [1, 1, categories])
    mapping_mat = np.expand_dims(mapping_mat, axis=0)
    mapped_prediction = prediction * mapping_mat
    mapped_prediction_with_minus = (1 - mapping_mat) * (-1e10) + mapped_prediction

    mapped_prediction = softmax(mapped_prediction_with_minus, axis=1)
    prediction_b2 = np.amax(mapped_prediction, axis=1)

    prediction = prediction_b1 * prediction_b2

    return prediction


prediction = np.array([[-4, -7, 0, 5],
                   [-3, 100, 10, -1],
                   [-2, 3, -8, -4],
                   [-5, 0, 9, -1]])

mapping_mat = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 1, 1, 0],
                        [0, 0, 1, 1]])

mapping_mat_fixed = category_mapping.complete_mapping_matrix(mapping_mat, 4)
print(mapping_mat_fixed)

print('Raw prediction: \n', prediction)

mapped_prediction = online_mapping(prediction, mapping_mat_fixed, 4)
print('Mapped prediction: \n', mapped_prediction)
