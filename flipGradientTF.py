import tensorflow as tf
from tensorflow.keras.layers import Layer


class GradientReversal(Layer):
    '''Flip the sign of gradient during training.'''
    def __init__(self, hp_lambda, reverse, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.hp_lambda = hp_lambda
        self.reverse = reverse

    def call(self, x):
        return self.grad_reverse(x)

    def get_config(self):
        config = {'hp_lambda': self.hp_lambda,
                  'reverse': self.reverse}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf.custom_gradient
    def grad_reverse(self, x):
        y = tf.identity(x)

        def custom_grad(dy):
            if self.reverse:
                return -dy * self.hp_lambda
            else:
                return dy * self.hp_lambda

        return y, custom_grad