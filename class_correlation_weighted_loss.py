from tensorflow.keras import backend as K
import tensorflow as tf

# Compatible with tensorflow backend
from evaluate_model import load_weights


def class_correlation_weighted_loss(gamma=2., alpha=.25, class_weights=None, power=1):

    if class_weights is None:
        weights_file = 'weights.csv'
        equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]
        classes, class_weights = load_weights(weights_file, equivalent_classes)

    class_weights = tf.constant(class_weights, dtype=tf.float32)

    def ccw_loss(y_true, y_pred):
        # clip to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        focal_weight = tf.where(tf.equal(y_true, 1), alpha * K.pow(1. - y_pred, gamma),
                          (1 - alpha) * K.pow(y_pred, gamma))

        y_true_normalized = y_true / tf.maximum(tf.reduce_sum(y_true, axis=1, keepdims=True),
                                                tf.ones_like(y_true[:, 0:1]))
        class_correlation_weight = 1 - tf.matmul(y_true_normalized, class_weights)
        class_correlation_weight = tf.maximum(y_true, class_correlation_weight)

        weight = focal_weight * K.pow(class_correlation_weight, power)

        return -K.mean(weight*(y_true * K.log(y_pred) + (1 - y_true) * K.log(1. - y_pred)))
    return ccw_loss