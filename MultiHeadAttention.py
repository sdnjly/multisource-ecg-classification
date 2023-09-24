import tensorflow as tf
from tensorflow.keras.layers import Dense, TimeDistributed, Dropout, Lambda, Activation, Add, Concatenate
import tensorflow.keras.backend as K
from tensorflow.keras import layers
import numpy as np

class MultiHeadAttention(layers.Layer):
    # mode 0 - big martixes, faster; mode 1 - more clear implementation
    def __init__(self, n_head, d_model, dropout, mode=0, **kwargs):
        self.mode = mode
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = self.d_v = d_k = d_v = d_model // n_head
        self.dropout = dropout
        self.supports_masking = True
        if mode == 0:
            self.qs_layer = Dense(n_head * d_k, use_bias=False)
            self.ks_layer = Dense(n_head * d_k, use_bias=False)
            self.vs_layer = Dense(n_head * d_v, use_bias=False)
        elif mode == 1:
            self.qs_layers = []
            self.ks_layers = []
            self.vs_layers = []
            for _ in range(n_head):
                self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
        self.attention = ScaledDotProductAttention()
        self.w_o = TimeDistributed(Dense(d_model))
        super().__init__(**kwargs)

    def call(self, inputs, mask=None):
        q, k, v = inputs[0], inputs[1], inputs[2]
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        head = None
        attn = None

        if self.mode == 0:
            qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
            ks = self.ks_layer(k)
            vs = self.vs_layer(v)

            def reshape1(x):
                s = tf.shape(x)  # [batch_size, len_q, n_head * d_k]
                x = tf.reshape(x, [s[0], s[1], n_head, s[2] // n_head])
                x = tf.transpose(x, [2, 0, 1, 3])
                x = tf.reshape(x, [-1, s[1], s[2] // n_head])  # [n_head * batch_size, len_q, d_k]
                return x

            qs = Lambda(reshape1)(qs)
            ks = Lambda(reshape1)(ks)
            vs = Lambda(reshape1)(vs)

            if mask is not None:
                mask = Lambda(lambda x: K.repeat_elements(x, n_head, 0))(mask)
            head = self.attention([qs, ks, vs], mask=mask)

            def reshape2(x):
                s = tf.shape(x)  # [n_head * batch_size, len_v, d_v]
                x = tf.reshape(x, [n_head, -1, s[1], s[2]])
                x = tf.transpose(x, [1, 2, 0, 3])
                x = tf.reshape(x, [-1, s[1], n_head * d_v])  # [batch_size, len_v, n_head * d_v]
                return x

            head = Lambda(reshape2)(head)
        elif self.mode == 1:
            heads = []

            for i in range(n_head):
                qs = self.qs_layers[i](q)
                ks = self.ks_layers[i](k)
                vs = self.vs_layers[i](v)
                head = self.attention([qs, ks, vs], mask=mask)
                heads.append(head)

            head = Concatenate()(heads) if n_head > 1 else heads[0]

        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], self.d_model


# It's safe to use a 1-d mask for self-attention
class ScaledDotProductAttention(layers.Layer):
    def __init__(self, attn_dropout=0.1, category_num=10, scale=False, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.dropout = attn_dropout
        self.scale = scale
        class_ids = tf.constant(np.expand_dims(np.arange(category_num), axis=0))
        self.q = K.cast(K.one_hot(K.cast(class_ids, dtype='uint8'), num_classes=category_num), dtype=K.floatx())

    def call(self, inputs, mask=None, training=True):  # mask_k or mask_qk
        k, v = inputs[0], inputs[1]
        if self.scale:
            temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype=K.floatx()))
        else:
            temper = 1
        attn = K.batch_dot(self.q, k, axes=[2, 2]) / temper  # shape=(batch, feature_len, seq_len)
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask, axis=1)
            attn = tf.math.multiply(attn, mask)
        attn = K.softmax(attn)
        if training:
            attn = K.dropout(attn, level=self.dropout)
        output = K.batch_dot(attn, v)
        return output

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][-1], input_shape[1][-1]


class ScaledDotProductAttention2(layers.Layer):
    def __init__(self, attn_dropout=0.1, scale=1, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.dropout = attn_dropout
        self.scale = scale

    def call(self, inputs, mask=None, training=True):  # mask_k or mask_qk
        q, k, v = inputs[0], inputs[1], inputs[2]
        attn = K.batch_dot(q, k, axes=[2, 2]) * self.scale  # shape=(batch, q, k)
        if mask is not None:
            mmask = (-1e+9) * (1. - K.cast(mask, K.floatx()))
            mmask = K.expand_dims(mmask, axis=1)
            attn = attn + mmask
        attn = K.softmax(attn)
        if training:
            attn = K.dropout(attn, level=self.dropout)
        output =  K.batch_dot(attn, v)
        return output

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], input_shape[2][-1]


class ScaledDotProductAttentionWithReference(layers.Layer):
    def __init__(self, attn_dropout=0.1, scale=False, reference_level=0, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.dropout = attn_dropout
        self.reference_level = reference_level
        self.scale = scale

    def call(self, inputs, mask=None, training=True):  # mask_k or mask_qk
        q, k, v = inputs[0], inputs[1], inputs[2]
        if self.scale:
            temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype=K.floatx()))
        else:
            temper = 1
        attn = K.batch_dot(q, k, axes=[2, 2]) / temper  # att shape=(batch, q_count, k_count)
        if mask is not None:
            mmask = (-1e+9) * (1. - K.cast(mask, K.floatx()))
            mmask = K.expand_dims(mmask, axis=1)
            attn = attn + mmask

        # Add reference point
        reference_features = K.zeros_like(v, dtype=K.floatx())[:,0:1,:]
        reference_attn = K.ones_like(q, dtype=K.floatx())[:,:,0:1] * self.reference_level
        v = K.concatenate([v, reference_features], axis=1)
        attn = K.concatenate([attn, reference_attn], axis=2)

        attn = K.softmax(attn, axis=-1)
        if training:
            attn = K.dropout(attn, self.dropout)
        output = K.batch_dot(attn, v)
        return output

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], input_shape[2][-1]


class ScaledDotProductAttentionWithReference2(layers.Layer):
    def __init__(self, attn_dropout=0.1, scale=1, reference_level=0, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.dropout = attn_dropout
        self.reference_level = reference_level
        self.scale = scale

    def call(self, inputs, mask=None):  # mask_k or mask_qk
        q, k, v = inputs[0], inputs[1], inputs[2]
        attn = K.batch_dot(q, k, axes=[2, 2]) * self.scale  # att shape=(batch, q_count, k_count)
        if mask is not None:
            mmask = (-1e+9) * (1. - K.cast(mask, K.floatx()))
            mmask = K.expand_dims(mmask, axis=1)
            attn = attn + mmask

        # Add reference point
        reference_features = K.zeros_like(v, dtype=K.floatx())[:,0:1,:]
        reference_attn = K.ones_like(q, dtype=K.floatx())[:,:,0:1] * self.reference_level
        v = K.concatenate([v, reference_features], axis=1)
        attn = K.concatenate([attn, reference_attn], axis=2)

        attn = K.softmax(attn, axis=-1)
        attn = K.dropout(attn, self.dropout)
        output = K.batch_dot(attn, v)
        return output

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], input_shape[2][-1]


class ScaledDotProductAttentionWithReferenceDrop(layers.Layer):
    def __init__(self, attn_dropout=0.1, scale=False, reference_level=0, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.dropout = attn_dropout
        self.reference_level = reference_level
        self.scale = scale

    def call(self, inputs, mask=None):  # mask_k or mask_qk
        q, k, v = inputs[0], inputs[1], inputs[2]
        if self.scale:
            temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype=K.floatx()))
        else:
            temper = 1
        attn = K.batch_dot(q, k, axes=[2, 2]) / temper  # att shape=(batch, q_count, k_count)
        if mask is not None:
            mmask = (-1e+9) * (1. - K.cast(mask, K.floatx()))
            mmask = K.expand_dims(mmask, axis=1)
            attn = attn + mmask

        # Add reference point
        reference_features = K.zeros_like(v, dtype=K.floatx())[:,0:1,:]
        reference_attn = K.ones_like(q, dtype=K.floatx())[:,:,0:1] * self.reference_level
        v = K.concatenate([v, reference_features], axis=1)
        attn = K.concatenate([attn, reference_attn], axis=2)

        attn_mask = K.sigmoid(K.relu(attn))
        attn = K.softmax(attn, axis=-1)
        attn = attn * attn_mask
        attn = K.dropout(attn, self.dropout)
        output = K.batch_dot(attn, v)
        return output

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], input_shape[2][-1]