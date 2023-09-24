import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import Layer


class WeightedGlobalAveragePooling1D(Layer):
    def __init__(self, a=0, **kwargs):
        self.supports_masking = True
        self.a = a
        super(WeightedGlobalAveragePooling1D, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        x = x[0]
        if mask is not None:
            # mask (batch, time)
            mask = K.cast(mask, K.floatx())
            # mask (batch, x_dim, time)
            mask = K.repeat(mask, x.shape[-1])
            # mask (batch, time, x_dim)
            mask = tf.transpose(mask, [0, 2, 1])
            # to make the masked values in x be equal to zero
            weighted_x = K.pow(x, self.a) * x * mask
            return K.sum(weighted_x, axis=1) / (K.sum(K.pow(x, self.a) * mask, axis=1) + K.epsilon())
        else:
            weighted_x = K.pow(x, self.a) * x
            return K.sum(weighted_x, axis=1) / (K.sum(K.pow(x, self.a), axis=1) + K.epsilon())

    def compute_output_shape(self, input_shape):
        # remove temporal dimension
        return input_shape[0][0], input_shape[0][2]
