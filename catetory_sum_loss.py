from tensorflow.keras import backend as K
import tensorflow as tf

# Compatible with tensorflow backend

def category_sum_loss():
    def inner_sum_loss(y_true, y_pred):
        return K.mean(K.maximum(y_pred - y_true, 0))
    return inner_sum_loss
