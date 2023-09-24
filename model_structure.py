# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 09:47:45 2017

@author: lenovo
"""
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras import backend as K
from tensorflow.python.keras.constraints import unit_norm
from HAN_attention import AttLayer_theano
from MultiHeadAttention import ScaledDotProductAttention
from WeightedGlobalAveragePooling1D import WeightedGlobalAveragePooling1D
from flipGradientTF import GradientReversal


def block1D_type1(x, nb_filter, filter_len, normalization_axis, dropout=0.5, kernel_regularizer=None,
                  kernel_initializer='glorot_uniform'):
    out = Conv1D(nb_filter, filter_len, padding='same',
                 kernel_regularizer=kernel_regularizer,
                 kernel_initializer=kernel_initializer)(x)
    out = BatchNormalization(axis=normalization_axis)(out)
    out = Activation('relu')(out)
    out = Dropout(dropout)(out)
    out = Conv1D(nb_filter, filter_len, padding='same',
                 kernel_regularizer=kernel_regularizer,
                 kernel_initializer=kernel_initializer)(out)
    # out = Add()([out,x],mode='sum')
    return out


def block1D_type2(x, nb_filter, filter_len, normalization_axis, dropout=0.5, kernel_regularizer=None,
                  kernel_initializer='glorot_uniform'):
    out = BatchNormalization(axis=normalization_axis)(x)
    out = Activation('relu')(out)
    out = Dropout(dropout)(out)
    out = Conv1D(nb_filter, filter_len, padding='same',
                 kernel_regularizer=kernel_regularizer,
                 kernel_initializer=kernel_initializer)(out)
    out = BatchNormalization(axis=normalization_axis)(out)
    out = Activation('relu')(out)
    out = Dropout(dropout)(out)
    out = Conv1D(nb_filter, filter_len, padding='same',
                 kernel_regularizer=kernel_regularizer,
                 kernel_initializer=kernel_initializer)(out)
    # out = Add()([out,x],mode='sum')
    return out


def SE_block(x, out_dim, ratio):
    squeeze = GlobalAveragePooling1D()(x)

    excitation = Dense(units=out_dim // ratio)(squeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=out_dim)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape((1, out_dim))(excitation)

    scale = multiply([x, excitation])

    return scale


def SE_block2(x, out_dim, ratio):
    excitation = Dense(units=out_dim // ratio)(x)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=out_dim)(excitation)
    excitation = Activation('sigmoid')(excitation)

    scale = multiply([x, excitation])

    return scale


def ngnet_part(input_name, categories, input_dimens, initfilters, initkernelsize,
               filters_increase=False, kernel_decrease=False,
               dropout=0.5, poolinglayers=6, normalization_axis=-1,
               kernel_initializer='he_normal', kernel_regularizer=None,
               filters_increase_interpools=2, kernel_decrease_interpools=3,
               conv_padding='same', multi_level_features=False):
    filters = int(initfilters)
    kernel_size = int(initkernelsize)

    feature_maps = []

    inp = Input(shape=input_dimens)
    hidden = Conv1D(filters, kernel_size, padding=conv_padding,
                    kernel_regularizer=kernel_regularizer,
                    kernel_initializer=kernel_initializer)(inp)  # also try kernel_initializer='he_uniform'
    hidden = BatchNormalization(axis=normalization_axis)(hidden)
    hidden_shortcut = Activation('relu')(hidden)

    hidden = block1D_type1(hidden_shortcut, filters, kernel_size,
                           normalization_axis, dropout,
                           kernel_regularizer,
                           kernel_initializer)
    hidden = MaxPooling1D(pool_size=2, padding='valid')(hidden)
    hidden_shortcut = MaxPooling1D(pool_size=2, padding='valid')(hidden_shortcut)
    merge_hidden = Add()([hidden, hidden_shortcut])
    if multi_level_features:
        feature_maps.append(Activation('relu')(BatchNormalization(axis=normalization_axis)(merge_hidden)))

    for i in range(1, poolinglayers):

        if filters_increase and i % filters_increase_interpools == 0:
            filters += int(initfilters)
            merge_hidden = Conv1D(filters, 1, padding=conv_padding,
                                  # kernel_initializer='he_normal',
                                  kernel_regularizer=kernel_regularizer)(merge_hidden)

        if kernel_decrease and i % kernel_decrease_interpools == 0:
            kernel_size = int(kernel_size / 2)
            if kernel_size < 2:
                kernel_size = int(2)

        hidden = block1D_type2(merge_hidden, filters,
                               kernel_size, normalization_axis, dropout,
                               kernel_regularizer, kernel_initializer)
        hidden = MaxPooling1D(pool_size=2, padding='valid')(hidden)
        merge_hidden = MaxPooling1D(pool_size=2, padding='valid')(merge_hidden)
        merge_hidden = Add()([hidden, merge_hidden])
        if multi_level_features:
            feature_maps.append(Activation('relu')(BatchNormalization(axis=normalization_axis)(merge_hidden)))

    if multi_level_features:
        out = feature_maps
    else:
        hidden = BatchNormalization(axis=normalization_axis)(merge_hidden)
        out = Activation('relu')(hidden)

    return inp, out


def ResNet_Attention(categories, ecg_length, ecg_part_name,
                     ecg_filters, ecg_kernelsize, ecg_filters_increase, ecg_kernel_decrease,
                     ecg_dropout, ecg_poolinglayers, dense_units=32, channels=1,
                     filters_increase_interpools=2, kernel_decrease_interpools=3,
                     atten_dim=32, l2_reg=1e-13):
    kernel_regularizer = l2(l2_reg)

    ecg_inp, ecg_features = ngnet_part(input_name=ecg_part_name,
                                       categories=categories,
                                       input_dimens=(ecg_length, channels),
                                       initfilters=ecg_filters, initkernelsize=ecg_kernelsize,
                                       filters_increase=ecg_filters_increase, kernel_decrease=ecg_kernel_decrease,
                                       dropout=ecg_dropout, poolinglayers=ecg_poolinglayers,
                                       kernel_regularizer=kernel_regularizer,
                                       filters_increase_interpools=filters_increase_interpools,
                                       kernel_decrease_interpools=kernel_decrease_interpools)

    features = AttLayer_theano(attention_dim=atten_dim)(ecg_features)
    features = Dense(dense_units, activation='relu', kernel_regularizer=kernel_regularizer)(features)
    prediction = Dense(categories, activation='sigmoid')(features)

    model = Model(inputs=[ecg_inp], outputs=[prediction])

    model.summary()

    return model


def ResNet_Attention_OnlineMapping(categories, mapping_matrix, ecg_length, ecg_part_name,
                                   ecg_filters, ecg_kernelsize, ecg_filters_increase, ecg_kernel_decrease,
                                   ecg_dropout, ecg_poolinglayers, dense_units=32, channels=1,
                                   filters_increase_interpools=2, kernel_decrease_interpools=3,
                                   atten_dim=32, l2_reg=1e-13, with_online_mapping=True):
    kernel_regularizer = l2(l2_reg)

    class_mask_inp = Input((categories,))

    ecg_inp, ecg_features = ngnet_part(input_name=ecg_part_name,
                                       categories=categories,
                                       input_dimens=(ecg_length, channels),
                                       initfilters=ecg_filters, initkernelsize=ecg_kernelsize,
                                       filters_increase=ecg_filters_increase, kernel_decrease=ecg_kernel_decrease,
                                       dropout=ecg_dropout, poolinglayers=ecg_poolinglayers,
                                       kernel_regularizer=kernel_regularizer,
                                       filters_increase_interpools=filters_increase_interpools,
                                       kernel_decrease_interpools=kernel_decrease_interpools)

    features = AttLayer_theano(attention_dim=atten_dim)(ecg_features)
    features = Dense(dense_units, activation='relu', kernel_regularizer=kernel_regularizer)(features)
    prediction = Dense(categories, activation='sigmoid')(features)

    # online mapping
    if with_online_mapping:
        prediction = tf.expand_dims(prediction, axis=1)
        prediction = tf.tile(prediction, tf.constant([1, categories, 1], tf.int32))
        mapping_matrix_tensor = tf.constant(mapping_matrix, tf.float32)
        mapping_matrix_tensor = tf.expand_dims(mapping_matrix_tensor, axis=0)
        mapped_prediction = Multiply()([prediction, mapping_matrix_tensor])
        mapped_prediction = GlobalMaxPool1D()(mapped_prediction)

        # masked prediction
        prediction = Multiply()([mapped_prediction, class_mask_inp])

    model = Model(inputs=[ecg_inp, class_mask_inp], outputs=[prediction])

    model.summary()

    return model


def ResNet_ClassWiseAttention(categories, atomic_categories, mapping_matrix, ecg_length, ecg_part_name,
                              ecg_filters, ecg_kernelsize, ecg_filters_increase, ecg_kernel_decrease,
                              ecg_dropout, ecg_poolinglayers, dense_units=32, channels=1,
                              filters_increase_interpools=2, kernel_decrease_interpools=3,
                              atten_dim=32, l2_reg=1e-13, with_online_mapping=True, with_class_mask=True,
                              with_feature_map_mask=False,
                              feature_map_length=None, domain_number=4, unit_norm_constraint=False,
                              key_regularization=False, group_sum_regularization=False):
    kernel_regularizer = l2(l2_reg)

    if with_class_mask:
        class_mask_inp = Input((categories,))
    else:
        class_mask_inp = None

    if with_feature_map_mask:
        sequence_mask_inp = Input((feature_map_length,))
    else:
        sequence_mask_inp = None

    ecg_inp, ecg_features = ngnet_part(input_name=ecg_part_name,
                                       categories=categories,
                                       input_dimens=(ecg_length, channels),
                                       initfilters=ecg_filters, initkernelsize=ecg_kernelsize,
                                       filters_increase=ecg_filters_increase, kernel_decrease=ecg_kernel_decrease,
                                       dropout=ecg_dropout, poolinglayers=ecg_poolinglayers,
                                       kernel_regularizer=kernel_regularizer,
                                       filters_increase_interpools=filters_increase_interpools,
                                       kernel_decrease_interpools=kernel_decrease_interpools)

    ecg_keys = Dense(categories, name='key_transform', activation='sigmoid')(ecg_features)
    features = ScaledDotProductAttention(attn_dropout=0.1, category_num=categories)([ecg_keys, ecg_features],
                                                                                    mask=sequence_mask_inp)

    predictions = []
    for i in range(atomic_categories):
        class_feature = Lambda(lambda x: x[:, i, :])(features)
        if unit_norm_constraint:
            class_prediction = Dense(1, activation='sigmoid', kernel_constraint=unit_norm(axis=0))(class_feature)
        else:
            class_prediction = Dense(1, activation='sigmoid')(class_feature)
        predictions.append(class_prediction)
    prediction = Concatenate(axis=-1)(predictions)

    # online mapping
    if with_online_mapping:
        mapping_matrix_tensor = tf.constant(mapping_matrix, tf.float32)
        if group_sum_regularization:
            prediction_group_sum = tf.linalg.matmul(prediction, mapping_matrix_tensor)
        else:
            prediction_group_sum = None
        prediction = tf.expand_dims(prediction, axis=1)
        prediction = tf.tile(prediction, tf.constant([1, categories, 1], tf.int32))
        mapping_matrix_tensor = tf.expand_dims(mapping_matrix_tensor, axis=0)
        mapped_prediction = Multiply()([prediction, mapping_matrix_tensor])
        mapped_prediction = GlobalMaxPool1D()(mapped_prediction)
        prediction = mapped_prediction

    # masked prediction
    if with_class_mask:
        prediction = Multiply()([prediction, class_mask_inp])
        if with_online_mapping and group_sum_regularization:
            prediction_group_sum = Multiply()([prediction_group_sum, class_mask_inp])

    # domain discriminator
    # domain_feature = WeightedGlobalAveragePooling1D()(ecg_features, mask=sequence_mask_inp)
    # domain_feature = GradientReversal(hp_lambda=1.0, reverse=True, name='gradient_reverse')(domain_feature)
    # domain_feature = Dense(dense_units, activation='relu', name='domain_layer1')(domain_feature)
    # domain_prediction = Dense(domain_number, activation='softmax', name='domain_layer2')(domain_feature)

    inputs = [ecg_inp]
    if with_class_mask:
        inputs.append(class_mask_inp)
    if with_feature_map_mask:
        inputs.append(sequence_mask_inp)

    outputs = [prediction]
    if key_regularization:
        outputs.append(ecg_keys)
    if with_online_mapping and group_sum_regularization:
        outputs.append(prediction_group_sum)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model


def ResNet_ClassWiseAttention_AtomicMapping(categories, atomic_categories, mapping_matrix, ecg_length, ecg_part_name,
                                            ecg_filters, ecg_kernelsize, ecg_filters_increase, ecg_kernel_decrease,
                                            ecg_dropout, ecg_poolinglayers, dense_units=32, channels=1,
                                            filters_increase_interpools=2, kernel_decrease_interpools=3,
                                            atten_dim=32, l2_reg=1e-13, with_online_mapping=True, with_class_mask=True,
                                            with_feature_map_mask=False,
                                            feature_map_length=None, domain_number=4, unit_norm_constraint=False,
                                            key_regularization=False, group_sum_regularization=False):
    kernel_regularizer = l2(l2_reg)

    if with_class_mask:
        class_mask_inp = Input((categories,))
    else:
        class_mask_inp = None

    if with_feature_map_mask:
        sequence_mask_inp = Input((feature_map_length,))
    else:
        sequence_mask_inp = None

    ecg_inp, ecg_features = ngnet_part(input_name=ecg_part_name,
                                       categories=categories,
                                       input_dimens=(ecg_length, channels),
                                       initfilters=ecg_filters, initkernelsize=ecg_kernelsize,
                                       filters_increase=ecg_filters_increase, kernel_decrease=ecg_kernel_decrease,
                                       dropout=ecg_dropout, poolinglayers=ecg_poolinglayers,
                                       kernel_regularizer=kernel_regularizer,
                                       filters_increase_interpools=filters_increase_interpools,
                                       kernel_decrease_interpools=kernel_decrease_interpools)

    if with_online_mapping:
        predicted_categories = atomic_categories
    else:
        predicted_categories = categories

    ecg_keys = Dense(predicted_categories, name='key_transform', activation='sigmoid')(ecg_features)
    features = ScaledDotProductAttention(attn_dropout=0.1, category_num=predicted_categories)([ecg_keys, ecg_features],
                                                                                              mask=sequence_mask_inp)
    predictions = []
    for i in range(predicted_categories):
        class_feature = Lambda(lambda x: x[:, i, :])(features)
        if unit_norm_constraint:
            class_prediction = Dense(1, kernel_constraint=unit_norm(axis=0))(class_feature)
        else:
            class_prediction = Dense(1, )(class_feature)
        predictions.append(class_prediction)
    prediction = Concatenate(axis=-1)(predictions)

    # online mapping
    if with_online_mapping:
        mapping_matrix_tensor = tf.constant(mapping_matrix, tf.float32)
        mapped_prediction = tf.linalg.matmul(prediction, mapping_matrix_tensor)
        prediction = Activation('sigmoid')(mapped_prediction)
    else:
        prediction = Activation('sigmoid')(prediction)

    # masked prediction
    if with_class_mask:
        prediction = Multiply()([prediction, class_mask_inp])

    # domain discriminator
    # domain_feature = WeightedGlobalAveragePooling1D()(ecg_features, mask=sequence_mask_inp)
    # domain_feature = GradientReversal(hp_lambda=1.0, reverse=True, name='gradient_reverse')(domain_feature)
    # domain_feature = Dense(dense_units, activation='relu', name='domain_layer1')(domain_feature)
    # domain_prediction = Dense(domain_number, activation='softmax', name='domain_layer2')(domain_feature)

    inputs = [ecg_inp]
    if with_class_mask:
        inputs.append(class_mask_inp)
    if with_feature_map_mask:
        inputs.append(sequence_mask_inp)

    outputs = [prediction]
    if key_regularization:
        outputs.append(ecg_keys)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model


def ResNet_ClassWiseAttention_SoftmaxMaximumMapping(categories, atomic_categories, mapping_matrix, ecg_length,
                                                    ecg_part_name,
                                                    ecg_filters, ecg_kernelsize, ecg_filters_increase,
                                                    ecg_kernel_decrease,
                                                    ecg_dropout, ecg_poolinglayers, dense_units=32, channels=1,
                                                    filters_increase_interpools=2, kernel_decrease_interpools=3,
                                                    atten_dim=32, l2_reg=1e-13, with_online_mapping=True,
                                                    online_mapping_type=None,
                                                    output_sum_mapping=False,
                                                    with_class_mask=True, with_feature_map_mask=False,
                                                    feature_map_length=None, domain_number=4,
                                                    unit_norm_constraint=False,
                                                    key_regularization=False, group_sum_regularization=False,
                                                    using_class_wise_atten=True,
                                                    using_global_max_pooling=False, using_multi_scale_features=False,
                                                    dense_regularizer=None, using_dense_SE=False):
    kernel_regularizer_l2 = l2(l2_reg)

    if with_class_mask:
        class_mask_inp = Input((categories,))
    else:
        class_mask_inp = None

    if with_feature_map_mask:
        sequence_mask_inp = Input((feature_map_length,))
    else:
        sequence_mask_inp = None

    ecg_inp, ecg_features = ngnet_part(input_name=ecg_part_name,
                                       categories=categories,
                                       input_dimens=(ecg_length, channels),
                                       initfilters=ecg_filters, initkernelsize=ecg_kernelsize,
                                       filters_increase=ecg_filters_increase, kernel_decrease=ecg_kernel_decrease,
                                       dropout=ecg_dropout, poolinglayers=ecg_poolinglayers,
                                       kernel_regularizer=kernel_regularizer_l2,
                                       filters_increase_interpools=filters_increase_interpools,
                                       kernel_decrease_interpools=kernel_decrease_interpools,
                                       multi_level_features=using_multi_scale_features)

    if using_class_wise_atten:
        # ecg_keys = Dense(categories, name='key_transform', activation='sigmoid')(ecg_features)
        if using_multi_scale_features:
            features_list = []
            for i in range(len(ecg_features)):
                ecg_keys = Conv1D(filters=categories, kernel_size=1, activation='sigmoid')(ecg_features[i])
                features = ScaledDotProductAttention(attn_dropout=0.1, category_num=categories)(
                    [ecg_keys, ecg_features[i]],
                    mask=sequence_mask_inp)
                features_list.append(features)
            features = Concatenate(axis=-1)(features_list)
        else:
            ecg_keys = Conv1D(filters=categories, kernel_size=1, name='key_transform', activation='sigmoid')(
                ecg_features)
            features = ScaledDotProductAttention(attn_dropout=0.1, category_num=categories)([ecg_keys, ecg_features],
                                                                                            mask=sequence_mask_inp)
    elif using_global_max_pooling:
        if using_multi_scale_features:
            features = []
            for i in range(len(ecg_features)):
                features.append(GlobalMaxPooling1D()(ecg_features[i]))
            features = Concatenate(axis=-1)(features)
        else:
            features = GlobalMaxPooling1D()(ecg_features)
    else:
        if using_multi_scale_features:
            features = []
            for i in range(len(ecg_features)):
                features.append(AttLayer_theano(attention_dim=atten_dim)(ecg_features[i]))
            features = Concatenate(axis=-1)(features)
        else:
            features = AttLayer_theano(attention_dim=atten_dim)(ecg_features)

    predictions = []
    for i in range(categories):
        if using_class_wise_atten:
            class_feature = Lambda(lambda x: x[:, i, :])(features)
        else:
            class_feature = features

        if using_dense_SE:
            class_feature = SE_block2(class_feature, K.int_shape(class_feature)[-1], 1)

        if unit_norm_constraint:
            class_prediction = Dense(1, kernel_constraint=unit_norm(axis=0), kernel_regularizer=dense_regularizer)(
                class_feature)
        else:
            class_prediction = Dense(1, kernel_regularizer=dense_regularizer)(class_feature)
        predictions.append(class_prediction)
    prediction = Concatenate(axis=-1)(predictions)

    # online mapping
    if with_online_mapping:
        if output_sum_mapping:
            prediction_sum = online_mapping_sum(prediction, mapping_matrix, using_clip=False)

        if online_mapping_type == 'sum':
            prediction = online_mapping_sum(prediction, mapping_matrix, using_clip=True)
        elif online_mapping_type == 'max':
            prediction = online_mapping_max(prediction, mapping_matrix, categories)
        elif online_mapping_type == 'sum-softmax':
            prediction = online_mapping_sum_softmax(prediction, mapping_matrix, categories)
        elif online_mapping_type == 'max-softmax':
            prediction = online_mapping_max_softmax(prediction, mapping_matrix, categories)
        else:
            raise Exception("Error, no valid type of online mapping is specified.")

    else:
        prediction = Activation('sigmoid')(prediction)

    # masked prediction
    if with_class_mask:
        prediction = Multiply()([prediction, class_mask_inp])

    # domain discriminator
    # domain_feature = WeightedGlobalAveragePooling1D()(ecg_features, mask=sequence_mask_inp)
    # domain_feature = GradientReversal(hp_lambda=1.0, reverse=True, name='gradient_reverse')(domain_feature)
    # domain_feature = Dense(dense_units, activation='relu', name='domain_layer1')(domain_feature)
    # domain_prediction = Dense(domain_number, activation='softmax', name='domain_layer2')(domain_feature)

    inputs = [ecg_inp]
    if with_class_mask:
        inputs.append(class_mask_inp)
    if with_feature_map_mask:
        inputs.append(sequence_mask_inp)

    outputs = [prediction]
    if key_regularization:
        outputs.append(ecg_keys)
    if with_online_mapping and output_sum_mapping:
        outputs.append(prediction_sum)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model


def ResNet_ClassWiseAttention_MultiScale(categories, atomic_categories, mapping_matrix, ecg_length, ecg_part_name,
                                         ecg_filters, ecg_kernelsize, ecg_filters_increase, ecg_kernel_decrease,
                                         ecg_dropout, ecg_poolinglayers, dense_units=32, channels=1,
                                         filters_increase_interpools=2, kernel_decrease_interpools=3,
                                         atten_dim=32, l2_reg=1e-13, with_online_mapping=True, online_mapping_type=None,
                                         output_sum_mapping=False,
                                         with_class_mask=True, with_feature_map_mask=False,
                                         feature_map_length=None, domain_number=4, unit_norm_constraint=False,
                                         key_regularization=False, group_sum_regularization=False,
                                         using_class_wise_atten=True,
                                         using_global_max_pooling=False, using_multi_scale_features=False,
                                         dense_regularizer=None, using_dense_SE=False):
    kernel_regularizer_l2 = l2(l2_reg)

    if with_class_mask:
        class_mask_inp = Input((categories,))
    else:
        class_mask_inp = None

    if with_feature_map_mask:
        sequence_mask_inp = Input((feature_map_length,))
    else:
        sequence_mask_inp = None

    ecg_inp, ecg_features = ngnet_part(input_name=ecg_part_name,
                                       categories=categories,
                                       input_dimens=(ecg_length, channels),
                                       initfilters=ecg_filters, initkernelsize=ecg_kernelsize,
                                       filters_increase=ecg_filters_increase, kernel_decrease=ecg_kernel_decrease,
                                       dropout=ecg_dropout, poolinglayers=ecg_poolinglayers,
                                       kernel_regularizer=kernel_regularizer_l2,
                                       filters_increase_interpools=filters_increase_interpools,
                                       kernel_decrease_interpools=kernel_decrease_interpools,
                                       multi_level_features=using_multi_scale_features)

    if using_multi_scale_features:
        # upsampling
        for i in range(1, len(ecg_features)):
            ecg_features[i] = UpSampling1D(2**i)(ecg_features[i])

        ecg_features = Concatenate(axis=-1)(ecg_features)

    if using_class_wise_atten:
        # ecg_keys = Dense(categories, name='key_transform', activation='sigmoid')(ecg_features)
        ecg_keys = Conv1D(filters=categories, kernel_size=1, name='key_transform', activation='sigmoid')(
            ecg_features)
        features = ScaledDotProductAttention(attn_dropout=0.1, category_num=categories)([ecg_keys, ecg_features],
                                                                                        mask=sequence_mask_inp)

    elif using_global_max_pooling:
        features = GlobalMaxPooling1D()(ecg_features)
    else:
        features = AttLayer_theano(attention_dim=atten_dim)(ecg_features)

    predictions = []
    for i in range(categories):
        if using_class_wise_atten:
            class_feature = Lambda(lambda x: x[:, i, :])(features)
        else:
            class_feature = features

        if using_dense_SE:
            class_feature = SE_block2(class_feature, K.int_shape(class_feature)[-1], 1)

        if unit_norm_constraint:
            class_prediction = Dense(1, kernel_constraint=unit_norm(axis=0), kernel_regularizer=dense_regularizer)(
                class_feature)
        else:
            class_prediction = Dense(1, kernel_regularizer=dense_regularizer)(class_feature)
        predictions.append(class_prediction)
    prediction = Concatenate(axis=-1)(predictions)

    # online mapping
    if with_online_mapping:
        if output_sum_mapping:
            prediction_sum = online_mapping_sum(prediction, mapping_matrix, using_clip=False)

        if online_mapping_type == 'sum':
            prediction = online_mapping_sum(prediction, mapping_matrix, using_clip=True)
        elif online_mapping_type == 'max':
            prediction = online_mapping_max(prediction, mapping_matrix, categories)
        elif online_mapping_type == 'sum-softmax':
            prediction = online_mapping_sum_softmax(prediction, mapping_matrix, categories)
        elif online_mapping_type == 'max-softmax':
            prediction = online_mapping_max_softmax(prediction, mapping_matrix, categories)
        else:
            raise Exception("Error, no valid type of online mapping is specified.")

    else:
        prediction = Activation('sigmoid')(prediction)

    # masked prediction
    if with_class_mask:
        prediction = Multiply()([prediction, class_mask_inp])

    # domain discriminator
    # domain_feature = WeightedGlobalAveragePooling1D()(ecg_features, mask=sequence_mask_inp)
    # domain_feature = GradientReversal(hp_lambda=1.0, reverse=True, name='gradient_reverse')(domain_feature)
    # domain_feature = Dense(dense_units, activation='relu', name='domain_layer1')(domain_feature)
    # domain_prediction = Dense(domain_number, activation='softmax', name='domain_layer2')(domain_feature)

    inputs = [ecg_inp]
    if with_class_mask:
        inputs.append(class_mask_inp)
    if with_feature_map_mask:
        inputs.append(sequence_mask_inp)

    outputs = [prediction]
    if key_regularization:
        outputs.append(ecg_keys)
    if with_online_mapping and output_sum_mapping:
        outputs.append(prediction_sum)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model


def online_mapping_sum(prediction, mapping_matrix, using_clip=False):
    prediction = Activation('sigmoid')(prediction)
    mapping_matrix_tensor = tf.constant(mapping_matrix, tf.float32)
    prediction = tf.linalg.matmul(prediction, mapping_matrix_tensor)
    if using_clip:
        prediction = K.clip(prediction, 0, 1)

    return prediction


def online_mapping_max(prediction, mapping_matrix, categories):
    prediction = Activation('sigmoid')(prediction)
    prediction = tf.expand_dims(prediction, axis=2)
    prediction = tf.tile(prediction, tf.constant([1, 1, categories], tf.int32))
    mapping_matrix_tensor = tf.constant(mapping_matrix, tf.float32)
    mapping_matrix_tensor = tf.expand_dims(mapping_matrix_tensor, axis=0)
    mapped_prediction = Multiply()([prediction, mapping_matrix_tensor])

    prediction = GlobalMaxPool1D()(mapped_prediction)

    return prediction


# cannot be used solely
def online_mapping_softmax(prediction, mapping_matrix, categories):
    # mapping by softmax and maximum
    prediction = tf.expand_dims(prediction, axis=2)
    prediction = tf.tile(prediction, tf.constant([1, 1, categories], tf.int32))
    mapping_matrix_tensor = tf.constant(mapping_matrix, tf.float32)
    mapping_matrix_tensor = tf.expand_dims(mapping_matrix_tensor, axis=0)
    mapped_prediction = Multiply()([prediction, mapping_matrix_tensor])
    mapped_prediction_with_minus = (1 - mapping_matrix_tensor) * (-1e10) + mapped_prediction

    mapped_prediction = Softmax(axis=1)(mapped_prediction_with_minus)
    prediction = GlobalMaxPool1D()(mapped_prediction)

    return prediction


def online_mapping_sum_softmax(prediction, mapping_matrix, categories):
    # branch 1: mapping by sigmoid and sum
    prediction_b1 = online_mapping_sum(prediction, mapping_matrix, using_clip=True)

    # branch 2: mapping by softmax and maximum
    prediction_b2 = online_mapping_softmax(prediction, mapping_matrix, categories)

    prediction = prediction_b1 * prediction_b2

    return prediction


def online_mapping_max_softmax(prediction, mapping_matrix, categories):
    # branch 1: mapping by sigmoid and maximum
    prediction_b1 = online_mapping_max(prediction, mapping_matrix, categories)

    # branch 2: mapping by softmax and maximum
    prediction_b2 = online_mapping_softmax(prediction, mapping_matrix, categories)

    prediction = prediction_b1 * prediction_b2

    return prediction


def online_mapping_sum_softmax2(prediction, mapping_matrix, categories, softmax_mapping=True):
    # branch 1: mapping by summary
    prediction_relu = Activation('relu')(prediction)
    mapping_matrix_tensor = tf.constant(mapping_matrix, tf.float32)
    mapped_prediction = tf.linalg.matmul(prediction_relu, mapping_matrix_tensor)
    # mapped_prediction = K.clip(mapped_prediction, 1e-10, 1)
    mapped_prediction = tf.where(tf.equal(mapped_prediction, 0), prediction, mapped_prediction)
    prediction_b1 = Activation('sigmoid')(mapped_prediction)

    # branch 2: mapping by softmax and maximum
    prediction = Activation('relu')(prediction)
    prediction = tf.expand_dims(prediction, axis=2)
    prediction = tf.tile(prediction, tf.constant([1, 1, categories], tf.int32))
    mapping_matrix_tensor = tf.expand_dims(mapping_matrix_tensor, axis=0)
    mapped_prediction = Multiply()([prediction, mapping_matrix_tensor])
    mapped_prediction_with_minus = (1 - mapping_matrix_tensor) * (-1e10) + mapped_prediction

    mapped_prediction = Softmax(axis=1)(mapped_prediction_with_minus)
    prediction_b2 = GlobalMaxPool1D()(mapped_prediction)

    if softmax_mapping:
        prediction = Multiply()([prediction_b1, prediction_b2])
    else:
        prediction = prediction_b1

    return prediction

# def ResNet_RSC(categories, mapping_matrix, ecg_length, ecg_part_name,
#               ecg_filters, ecg_kernelsize, ecg_filters_increase, ecg_kernel_decrease,
#               ecg_dropout, ecg_poolinglayers, dense_units=32, channels=1,
#               filters_increase_interpools=2, kernel_decrease_interpools=3,
#               atten_dim=32, l2_reg=1e-13, with_online_mapping=True, feature_map_length=None,
#               domain_number=4, unit_norm_constraint=False):
#     kernel_regularizer = l2(l2_reg)
#
#     # Backbone
#     ecg_inp, ecg_features = ngnet_part(input_name=ecg_part_name,
#                                        categories=categories,
#                                        input_dimens=(ecg_length, channels),
#                                        initfilters=ecg_filters, initkernelsize=ecg_kernelsize,
#                                        filters_increase=ecg_filters_increase, kernel_decrease=ecg_kernel_decrease,
#                                        dropout=ecg_dropout, poolinglayers=ecg_poolinglayers,
#                                        kernel_regularizer=kernel_regularizer,
#                                        filters_increase_interpools=filters_increase_interpools,
#                                        kernel_decrease_interpools=kernel_decrease_interpools)
#     backbone = Model(inputs=[ecg_inp], outputs=[ecg_features])
#
#     # classifier
#     class_head = tf.keras.models.Sequential([tf.keras.layers.Dense(dense_units, activation='relu'),
#                                              tf.keras.layers.Dense(categories)])  # no sigmoid
#
#     model = RSCModelWrapper(backbone, class_head,
#                             trainable_backbone=True,
#                             percentile=66, batch_percentage=33)
#
#     return model
