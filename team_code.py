#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of required functions, remove non-required functions, and add your own function.
import csv
from copy import deepcopy
from datetime import datetime

import joblib
import math
import tensorflow as tf
from scipy.signal import butter, filtfilt
from sklearn.metrics import precision_recall_curve
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from tensorflow.keras.regularizers import l1, l2

import model_structure
import smooth
from KMM import KMM
from L1_loss import l1_loss
from catetory_sum_loss import category_sum_loss
from class_correlation_weighted_loss import class_correlation_weighted_loss
from data_prepare import weighted_batch_generator, load_data2, load_data3
from evaluate_model import evaluate_model, load_weights, get_category_field, replace_equivalent_classes, arrange_labels, \
    load_weights2
from focal_loss import focal_loss_2
from helper_code import *
import category_mapping

twelve_lead_model_filename = '12_lead_model'
six_lead_model_filename = '6_lead_model'
three_lead_model_filename = '3_lead_model'
two_lead_model_filename = '2_lead_model'

# Define the Challenge lead sets. These variables are not required. You can change or remove them.
twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
six_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF')
four_leads = ('I', 'II', 'III', 'V2')
three_leads = ('I', 'II', 'V2')
two_leads = ('I', 'II')
lead_sets = (twelve_leads, six_leads, four_leads, three_leads, two_leads)

################################################################################
#
# Training function
#
################################################################################

# Train your model. This function is *required*. Do *not* change the arguments of this function.
def training_code(data_directory, train_paths, test_path, test_set_name, model_directory, use_offlinemapping=True,
                  use_onlinemapping=True, use_class_mask=True, online_mapping_type=None, hirarchical_based_mask=True,
                  key_regularization=False, group_sum_regularization=False, evaluate_only_scored_classes=False,
                  using_class_wise_atten=True, using_global_max_pooling=False, ecg_poolinglayers=7,
                  using_multi_scale_features=False, dense_regularization_type='l2', using_dense_SE=False,
                  output_sum_mapping=False, b=0.3, ):
    # Find header and recording files.
    print('Finding header and recording files...')

    # category_map = category_mapping.get_label_mapping_PTBXL()
    category_map = category_mapping.get_label_mapping()

    # Extract classes from dataset.
    print('Extracting classes...')

    equivalent_classes = [['284470004', '63593006'], ['427172004', '17338001', '164884008']]
    # equivalent_classes = [['427172004', '17338001', '164884008']]
    base_classes = set(category_map.keys())
    for sub_cates in category_map.values():
        base_classes |= set(sub_cates)
    classes = load_classes(data_directory, base_classes)
    classes = sorted(classes)
    classes = replace_equivalent_classes(classes, equivalent_classes)

    # compute mapping matrix
    mapping_matrix = category_mapping.get_mapping_mat(classes, category_map)
    # complete the mapping matrix
    mapping_matrix = category_mapping.complete_mapping_matrix(mapping_matrix, deepth=4)

    # Extract features and labels from dataset.
    print('Extracting data and labels...')
    target_fs = 250
    target_length = 5120

    train_data_list = []
    train_label_list = []
    train_rawlabel_list = []
    train_classmask_list = []
    train_feamapmask_list = []
    train_domainid_list = []

    for i in range(len(train_paths)):
        print(f'loading {train_paths[i]}')
        data, labels, raw_labels, class_mask, feature_map_mask = \
            load_data3(train_paths[i],
                       target_fs=target_fs,
                       target_length=target_length,
                       mask_length=39,
                       classes=classes, category_mapping_mat=mapping_matrix,
                       leads=twelve_leads,
                       maskout_unmapped=hirarchical_based_mask,
                       using_offline_mapping=use_offlinemapping)
        domain_id = np.ones((len(data), 1), dtype=np.int32) * i

        train_data_list.append(data)
        train_label_list.append(labels)
        train_rawlabel_list.append(raw_labels)
        train_classmask_list.append(class_mask)
        train_feamapmask_list.append(feature_map_mask)
        train_domainid_list.append(domain_id)

        print(f'data min {data.min()}')
        print(f'data max {data.max()}')
        print(f'class_mask min {class_mask.min()}')
        print(f'class_mask max {class_mask.max()}')
        print(f'labels min {labels.min()}')
        print(f'labels max {labels.max()}')


    # use G12EC as target
    X_train = np.concatenate(train_data_list, axis=0)
    Y_train = np.concatenate(train_label_list, axis=0)
    mask_train = np.concatenate(train_classmask_list, axis=0)
    feature_map_mask_train = np.concatenate(train_feamapmask_list, axis=0)
    domain_id_train = np.concatenate(train_domainid_list, axis=0)

    X_test, Y_test, labels_test, mask_test, feature_map_mask_test = \
        load_data3(test_path,
                   target_fs=target_fs,
                   target_length=target_length,
                   mask_length=39,
                   classes=classes, category_mapping_mat=mapping_matrix,
                   leads=twelve_leads,
                   maskout_unmapped=hirarchical_based_mask,
                   using_offline_mapping=True)

    # valid labels
    labels_count_train = Y_train.sum(axis=0)
    labels_count_test = Y_test.sum(axis=0)
    valid_label_ids = np.where(np.logical_and(labels_count_train > 50, labels_count_test > 10))[0]
    # valid_label_ids = np.where(labels_count_train > 50)[0]
    Y_test = Y_test[:, valid_label_ids]
    Y_train = Y_train[:, valid_label_ids]
    mask_train = mask_train[:, valid_label_ids]
    mask_test = mask_test[:, valid_label_ids]
    valid_classes = [classes[i] for i in valid_label_ids]
    mapping_matrix = mapping_matrix[valid_label_ids, :]
    mapping_matrix = mapping_matrix[:, valid_label_ids]

    num_classes = len(valid_classes)

    # shuffle
    [X_train, Y_train, mask_train, feature_map_mask_train, domain_id_train], positions = \
        unison_shuffled_copies([X_train, Y_train,
                                mask_train,
                                feature_map_mask_train,
                                domain_id_train])

    # kmm = KMM(kernel_type='linear', B=20)
    # sample_weights = kmm.fit_in_batches(Y_train, Y_test, 10000)
    # sample_weights = sample_weights.squeeze()
    # sample_weights = np.float64(sample_weights) / sample_weights.sum()
    sample_weights = None

    l1_reg = 1e-6
    l2_reg = 1e-8
    if dense_regularization_type == 'l1':
        dense_regularizer = l1(l1_reg)
    elif dense_regularization_type == 'l2':
        dense_regularizer = l2(l2_reg)
    else:
        dense_regularizer = None
    # Define parameters for DNN model.
    params = {'categories': num_classes, 'atomic_categories': num_classes, 'mapping_matrix': mapping_matrix,
              'ecg_length': target_length, 'ecg_part_name': 'ecg',
              'ecg_filters': 32, 'ecg_kernelsize': 16, 'ecg_filters_increase': False, 'ecg_kernel_decrease': False,
              'ecg_dropout': 0.25, 'ecg_poolinglayers': ecg_poolinglayers, 'channels': 12,
              'filters_increase_interpools': 3, 'kernel_decrease_interpools': 3, 'l2_reg': l2_reg,
              'key_regularization': key_regularization, 'group_sum_regularization': group_sum_regularization,
              'with_online_mapping': use_onlinemapping, 'online_mapping_type': online_mapping_type,
              'using_class_wise_atten': using_class_wise_atten,
              'using_global_max_pooling': using_global_max_pooling,
              'using_multi_scale_features': using_multi_scale_features, 'dense_regularizer': dense_regularizer,
              'using_dense_SE': using_dense_SE, 'output_sum_mapping': output_sum_mapping}

    if using_class_wise_atten:
        model_name = f'ResNet-ClassWiseAtten'
    elif using_global_max_pooling:
        model_name = f'ResNet-GMP'
    else:
        model_name = f'ResNet-Atten'

    labelmask_name = "label-mask-full" if hirarchical_based_mask else "label-mask-uncovered"

    # Train models.
    # Create a folder for the model if it does not already exist.
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

    print('Training ECG model...')

    params['with_online_mapping'] = False
    params['with_class_mask'] = False
    params['using_multi_scale_features'] = False
    params['using_dense_SE'] = False
    test_name = test_set_name + "-baseline"
    train_model(params, model_directory, X_train, Y_train, mask_train, feature_map_mask_train, domain_id_train,
                X_test, Y_test, mask_test, feature_map_mask_test, labels_test,
                valid_classes, twelve_leads, target_fs, target_length, model_name, test_name, epochs=100,
                time_stamp='20220118-045257', evaluate_only_scored_classes=evaluate_only_scored_classes,
                sample_weights=sample_weights, b=b)

    params['with_online_mapping'] = False
    params['with_class_mask'] = False
    params['using_multi_scale_features'] = True
    params['using_dense_SE'] = False
    test_name = test_set_name +"_multiscale" + "-baseline"
    train_model(params, model_directory, X_train, Y_train, mask_train, feature_map_mask_train, domain_id_train,
                X_test, Y_test, mask_test, feature_map_mask_test, labels_test,
                valid_classes, twelve_leads, target_fs, target_length, model_name, test_name, epochs=100,
                time_stamp='20220118-024131', evaluate_only_scored_classes=evaluate_only_scored_classes,
                sample_weights=sample_weights, b=b)

    params['with_online_mapping'] = False
    params['with_class_mask'] = False
    params['using_multi_scale_features'] = True
    params['using_dense_SE'] = True
    test_name = test_set_name + "_multiscale_SE" + "-baseline"
    train_model(params, model_directory, X_train, Y_train, mask_train, feature_map_mask_train, domain_id_train,
                X_test, Y_test, mask_test, feature_map_mask_test, labels_test,
                valid_classes, twelve_leads, target_fs, target_length, model_name, test_name, epochs=100,
                time_stamp='20220117-224252', evaluate_only_scored_classes=evaluate_only_scored_classes,
                sample_weights=sample_weights, b=b)

    # params['with_online_mapping'] = False
    # params['with_class_mask'] = False
    # params['using_multi_scale_features'] = False
    # params['using_dense_SE'] = False
    # test_name = test_set_name + '-' + online_mapping_type
    # train_model(params, model_directory, X_train, Y_train, mask_train, feature_map_mask_train, domain_id_train,
    #             X_test, Y_test, mask_test, feature_map_mask_test, labels_test,
    #             valid_classes, twelve_leads, target_fs, target_length, model_name, test_name, epochs=100,
    #             time_stamp=None, evaluate_only_scored_classes=evaluate_only_scored_classes,
    #             sample_weights=sample_weights, b=b)
    #
    # params['with_online_mapping'] = True
    # params['with_class_mask'] = True
    # params['using_multi_scale_features'] = True
    # params['using_dense_SE'] = True
    # params['mapping_matrix'] = category_mapping.get_mapping_mat(valid_classes, category_map)
    # test_name = test_set_name + "_multiscale_SE" + '-' + labelmask_name + '-' + online_mapping_type
    # train_model(params, model_directory, X_train, Y_train, mask_train, feature_map_mask_train, domain_id_train,
    #             X_test, Y_test, mask_test, feature_map_mask_test, labels_test,
    #             valid_classes, twelve_leads, target_fs, target_length, model_name, test_name, epochs=100,
    #             time_stamp='20220117-204811', evaluate_only_scored_classes=evaluate_only_scored_classes,
    #             sample_weights=sample_weights, b=b)
    #
    # params['with_online_mapping'] = False
    # params['with_class_mask'] = True
    # test_name = test_set_name + '-' + labelmask_name
    # train_model(params, model_directory, X_train, Y_train, mask_train, feature_map_mask_train, domain_id_train,
    #             X_test, Y_test, mask_test, feature_map_mask_test, labels_test,
    #             valid_classes, twelve_leads, target_fs, target_length, model_name, test_name, epochs=100,
    #             time_stamp='20220116-181023', evaluate_only_scored_classes=evaluate_only_scored_classes,
    #             sample_weights=sample_weights, b=b)

    # params['with_online_mapping'] = False
    # params['with_class_mask'] = False
    # params['using_multi_scale_features'] = False
    # params['using_dense_SE'] = False
    # test_name = test_set_name + "-baseline"
    # train_model(params, model_directory, X_train, Y_train, mask_train, feature_map_mask_train, domain_id_train,
    #             X_test, Y_test, mask_test, feature_map_mask_test, labels_test,
    #             valid_classes, twelve_leads, target_fs, target_length, model_name, test_name, epochs=100,
    #             time_stamp=None, evaluate_only_scored_classes=evaluate_only_scored_classes,
    #             sample_weights=sample_weights, b=b)

    # params['with_online_mapping'] = False
    # params['with_class_mask'] = False
    # params['using_multi_scale_features'] = True
    # params['using_dense_SE'] = False
    # test_name = test_set_name + "_multi_scale"
    # train_model(params, model_directory, X_train, Y_train, mask_train, feature_map_mask_train, domain_id_train,
    #             X_test, Y_test, mask_test, feature_map_mask_test, labels_test,
    #             valid_classes, twelve_leads, target_fs, target_length, model_name, test_name, epochs=100,
    #             time_stamp=None, evaluate_only_scored_classes=evaluate_only_scored_classes,
    #             sample_weights=sample_weights, b=b)
    #
    #
    # params['with_online_mapping'] = False
    # params['with_class_mask'] = False
    # params['using_multi_scale_features'] = True
    # params['using_dense_SE'] = True
    # test_name = test_set_name + "_multi_scale_SE"
    # train_model(params, model_directory, X_train, Y_train, mask_train, feature_map_mask_train, domain_id_train,
    #             X_test, Y_test, mask_test, feature_map_mask_test, labels_test,
    #             valid_classes, twelve_leads, target_fs, target_length, model_name, test_name, epochs=100,
    #             time_stamp=None, evaluate_only_scored_classes=evaluate_only_scored_classes,
    #             sample_weights=sample_weights, b=b)


def cross_validation_with_simulated_db_difference(data_directory, model_directory,  key_regularization=False,
                  group_sum_regularization=False):

    # Extract classes.
    print('Extracting classes...')
    category_map = category_mapping.get_label_mapping2()
    base_classes = set(category_map.keys())
    for sub_cates in category_map.values():
        base_classes |= set(sub_cates)

    classes = sorted(base_classes)

    # compute mapping matrix
    mapping_matrix = category_mapping.get_mapping_mat(classes, category_map)
    # atomic_mapping_matrix = category_mapping.get_mapping_mat_of_atomic_types(mapping_matrix)
    atomic_mapping_matrix = mapping_matrix
    num_classes = atomic_mapping_matrix.shape[1]
    num_atomic_classes = atomic_mapping_matrix.shape[0]

    # load data
    print('Extracting data and labels...')
    target_fs = 250
    target_length = 5000

    data, labels, raw_labels, class_mask, feature_map_mask = \
        load_data(data_directory,
                  target_fs=target_fs,
                  target_length=target_length,
                  mask_length=39,
                  classes=classes, category_mapping_mat=mapping_matrix,
                  leads=twelve_leads)

    print('data shape:', data.shape)

    # split to subsets
    fold_number = 5
    randomize = np.arange(len(data))
    np.random.shuffle(randomize)
    data = data[randomize]
    labels = labels[randomize]
    class_mask = class_mask[randomize]
    fold_size = int(len(data)/fold_number)
    X_folds, Y_folds, class_mask_folds = [], [], []
    for i in range(fold_number):
        X_folds.append(data[i*fold_size: (i+1)*fold_size])
        Y_folds.append(labels[i*fold_size: (i+1)*fold_size])
        class_mask_folds.append(class_mask[i*fold_size: (i+1)*fold_size])

    # simulate annotation differences among datasets
    masked_categories = [{'59118001':['713427006'], # RBBB: CRBBB;
                          '164909002':['733534002'],# LBBB: CLBBB
                          '55930002': ['164930006', '429622005']},  # STC: STIAb, STD,
                         {'59118001': ['713426002', '713427006'],   # RBBB: IRBBB, CRBBB
                          '195039008': ['270492004', '195042002'],  # PAB: IAVB, IIAVB
                          '55930002': ['164930006']},  # STC: STIAb
                         {'164909002': ['445211001', '445118002'], # LBBB: LPFB, LAnFB,
                          '195042002': ['426183003', '54016002'],  # IIAVB: IIAVBII, MoI
                          '55930002':  ['429622005'], # STC: STD
                          '365418004': ['59931005']}, # T wave findings: TInV
                         {'164909002': ['251120003', '733534002'], # LBBB: ILBBB, CLBBB
                          '233917008': ['195039008'],  # AVB: PAB
                          '195039008': ['270492004', '195042002']}, # PAB: IAVB, IIAVB
                         {'164909002': ['445211001', '445118002', '251120003', '733534002'], # LBBB: LPFB, LAnFB, ILBBB, CLBBB
                          '233917008': ['27885002'], # AVB: CHB
                          '365418004': ['164934002', '59931005']}]  # T wave findings: TAb, TInV
    Y_folds_modified = []
    class_mask_folds_modified = []
    for i in range(fold_number):
        Y_fold_, class_mask_ = category_mapping.modify_label_level(Y_folds[i], class_mask_folds[i], masked_categories[i], classes)
        Y_folds_modified.append(Y_fold_)
        class_mask_folds_modified.append(class_mask_)

    # Define parameters for DNN model.
    params = {
        'categories': num_classes,
        'atomic_categories': num_atomic_classes,
        'mapping_matrix': atomic_mapping_matrix,
        'ecg_length': target_length,
        'ecg_part_name': 'ecg',
        'ecg_filters': 32,
        'ecg_kernelsize': 16,
        'ecg_filters_increase': True,
        'ecg_kernel_decrease': True,
        'ecg_dropout': 0.2,
        'ecg_poolinglayers': 7,
        'channels': 12,
        'filters_increase_interpools': 3,
        'kernel_decrease_interpools': 3,
        'l2_reg': 1e-8,
        'key_regularization': key_regularization,
        'group_sum_regularization': group_sum_regularization,
    }

    # cross validation
    for i in range(fold_number):

        X_test, Y_test = X_folds[i], Y_folds[i]
        Y_test = category_mapping.offline_mapping(Y_test, mapping_matrix, deepth=4)
        class_mask_test = class_mask_folds[i]

        X_train, Y_train = [], []
        for j in range(fold_number):
            if j != i:
                X_train.append(X_folds[j])
                Y_train.append(Y_folds[j])
        X_train = np.concatenate(X_train, axis=0)
        Y_train = np.concatenate(Y_train, axis=0)
        Y_train = category_mapping.offline_mapping(Y_train, mapping_matrix, deepth=4)

        model_name = f'ResNet-ClassWiseAtten-SingleDB-fold{i}'
        # train model without annotation modifications
        params['with_online_mapping'] = False
        params['with_class_mask'] = False
        train_model(params, model_directory, X_train, Y_train, None, None, None,
                    X_test, Y_test, None, None, None,
                    classes, twelve_leads, target_fs, target_length, model_name, "OriginalTrainingSet", epochs=100)

        # modified labels
        Y_train_modified = []
        class_mask_train_modified = []
        for j in range(fold_number):
            if j != i:
                Y_train_modified.append(Y_folds_modified[j])
                class_mask_train_modified.append(class_mask_folds_modified[j])
        Y_train_modified = np.concatenate(Y_train_modified, axis=0)
        # offline mapping
        Y_train_modified = category_mapping.offline_mapping(Y_train_modified, mapping_matrix, deepth=4)

        class_mask_train_modified = np.concatenate(class_mask_train_modified, axis=0)

        # train model with annotation modifications, but without online mapping and label mask
        params['with_online_mapping'] = False
        params['with_class_mask'] = False
        train_model(params, model_directory, X_train, Y_train_modified, None, None, None,
                    X_test, Y_test, None, None, None,
                    classes, twelve_leads, target_fs, target_length, model_name, "ModifiedTrainingSet", epochs=100)

        # train model with annotation modifications, and with online mapping and label mask
        params['with_online_mapping'] = True
        params['with_class_mask'] = True
        train_model(params, model_directory, X_train, Y_train_modified, class_mask_train_modified, None, None,
                    X_test, Y_test, class_mask_test, None, None,
                    classes, twelve_leads, target_fs, target_length, model_name, "ModifiedTrainingSet_OnlineMapping", epochs=100)


def cross_validation_with_simulated_db_difference_2(data_directory, model_directory,  key_regularization=False,
                  group_sum_regularization=False, masked_categories=None):

    # Extract classes.
    print('Extracting classes...')
    category_map = category_mapping.get_label_mapping2()
    base_classes = set(category_map.keys())
    for sub_cates in category_map.values():
        base_classes |= set(sub_cates)

    classes = sorted(base_classes)

    # compute mapping matrix
    mapping_matrix = category_mapping.get_mapping_mat(classes, category_map)
    # atomic_mapping_matrix = category_mapping.get_mapping_mat_of_atomic_types(mapping_matrix)
    atomic_mapping_matrix = mapping_matrix
    num_classes = atomic_mapping_matrix.shape[1]
    num_atomic_classes = atomic_mapping_matrix.shape[0]

    # load data
    print('Extracting data and labels...')
    target_fs = 250
    target_length = 5000

    data, labels, raw_labels, class_mask, feature_map_mask = \
        load_data(data_directory,
                  target_fs=target_fs,
                  target_length=target_length,
                  mask_length=39,
                  classes=classes, category_mapping_mat=mapping_matrix,
                  leads=twelve_leads)

    print('data shape:', data.shape)

    # split to subsets
    fold_number = 5
    randomize = np.arange(len(data))
    np.random.shuffle(randomize)
    data = data[randomize]
    labels = labels[randomize]
    class_mask = class_mask[randomize]
    fold_size = int(len(data)/fold_number)
    X_folds, Y_folds, class_mask_folds = [], [], []
    for i in range(fold_number):
        X_folds.append(data[i*fold_size: (i+1)*fold_size])
        Y_folds.append(labels[i*fold_size: (i+1)*fold_size])
        class_mask_folds.append(class_mask[i*fold_size: (i+1)*fold_size])

    Y_folds_modified = []
    class_mask_folds_modified = []
    for i in range(fold_number):
        Y_fold_, class_mask_ = category_mapping.modify_label_level(Y_folds[i], class_mask_folds[i], masked_categories[i], classes)
        Y_folds_modified.append(Y_fold_)
        class_mask_folds_modified.append(class_mask_)

    # Define parameters for DNN model.
    params = {
        'categories': num_classes,
        'atomic_categories': num_atomic_classes,
        'mapping_matrix': atomic_mapping_matrix,
        'ecg_length': target_length,
        'ecg_part_name': 'ecg',
        'ecg_filters': 32,
        'ecg_kernelsize': 16,
        'ecg_filters_increase': True,
        'ecg_kernel_decrease': True,
        'ecg_dropout': 0.2,
        'ecg_poolinglayers': 7,
        'channels': 12,
        'filters_increase_interpools': 3,
        'kernel_decrease_interpools': 3,
        'l2_reg': 1e-8,
        'key_regularization': key_regularization,
        'group_sum_regularization': group_sum_regularization,
    }

    # cross validation
    for i in range(fold_number):

        X_test, Y_test = X_folds[i], Y_folds[i]
        Y_test = category_mapping.offline_mapping(Y_test, mapping_matrix, deepth=4)
        class_mask_test = class_mask_folds[i]

        X_train, Y_train = [], []
        for j in range(fold_number):
            if j != i:
                X_train.append(X_folds[j])
                Y_train.append(Y_folds[j])
        X_train = np.concatenate(X_train, axis=0)
        Y_train = np.concatenate(Y_train, axis=0)
        Y_train = category_mapping.offline_mapping(Y_train, mapping_matrix, deepth=4)

        model_name = f'ResNet-ClassWiseAtten-SingleDB-fold{i}'
        # train model without annotation modifications
        params['with_online_mapping'] = False
        params['with_class_mask'] = False
        train_model(params, model_directory, X_train, Y_train, None, None, None,
                    X_test, Y_test, None, None, None,
                    classes, twelve_leads, target_fs, target_length, model_name, "OriginalTrainingSet", epochs=100)

        # modified labels
        Y_train_modified = []
        class_mask_train_modified = []
        for j in range(fold_number):
            if j != i:
                Y_train_modified.append(Y_folds_modified[j])
                class_mask_train_modified.append(class_mask_folds_modified[j])
        Y_train_modified = np.concatenate(Y_train_modified, axis=0)
        # offline mapping
        Y_train_modified = category_mapping.offline_mapping(Y_train_modified, mapping_matrix, deepth=4)

        class_mask_train_modified = np.concatenate(class_mask_train_modified, axis=0)

        # train model with annotation modifications, but without online mapping and label mask
        params['with_online_mapping'] = False
        params['with_class_mask'] = False
        train_model(params, model_directory, X_train, Y_train_modified, None, None, None,
                    X_test, Y_test, None, None, None,
                    classes, twelve_leads, target_fs, target_length, model_name, "ModifiedTrainingSet", epochs=100)

        # train model with annotation modifications, and with online mapping and label mask
        params['with_online_mapping'] = True
        params['with_class_mask'] = True
        train_model(params, model_directory, X_train, Y_train_modified, class_mask_train_modified, None, None,
                    X_test, Y_test, class_mask_test, None, None,
                    classes, twelve_leads, target_fs, target_length, model_name, "ModifiedTrainingSet_OnlineMapping", epochs=100)


def load_classes(data_directory, base_classes):
    header_files, recording_files = find_challenge_files(data_directory)

    classes = set(base_classes)
    for header_file in header_files:
        header = load_header(header_file)
        classes |= set(get_labels(header))

    classes -= {''}
    if all(is_integer(x) for x in classes):
        classes = sorted(classes, key=lambda x: int(x))  # Sort classes numerically if numbers.
    else:
        classes = sorted(classes)  # Sort classes alphanumerically if not numbers.

    return classes


def load_data(data_directory, target_fs=250, target_length=5000, mask_length=39, classes=None,
              category_mapping_mat=None, leads=None):
    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)
    num_classes = len(classes)

    if not num_recordings:
        raise Exception('No data was provided.')

    data = np.zeros((num_recordings, target_length, 12), dtype='float32')
    labels = np.zeros((num_recordings, num_classes), dtype='float32')  # One-hot encoding of classes
    feature_map_mask = np.ones((num_recordings, mask_length), dtype='float32')
    raw_labels = []

    for i in range(num_recordings):
        # print('    {}/{}...'.format(i + 1, num_recordings))

        # Load header and recording.
        header = load_header(header_files[i])
        recording = load_recording(recording_files[i], header, leads)
        recording = recording.transpose()

        # get sampling frequency of the file
        header_data = header.split('\n')
        fs = int(header_data[0].split(' ')[2])
        original_length = int(header_data[0].split(' ')[3])

        # get resolution
        rs = int(header_data[1].split(' ')[2].split('/')[0])
        recording = recording / rs

        if fs != target_fs:
            step = round(fs / target_fs)
            signal_length = recording.shape[0]
            recording = recording[0:signal_length:step, :]

        # remove baseline wander
        for j in range(recording.shape[1]):
            smoothed_signal = smooth.smooth(recording[:, j], window_len=target_fs, window='flat')
            recording[:, j] = recording[:, j] - smoothed_signal

        # band pass filtering
        # recording = bandpass_filter(recording, 0.1, 30, target_fs, 1)

        # normalization
        # scaler = StandardScaler()
        # scaler.fit(recording[:, 1:2])  # fit the signal in Lead II
        # for j in range(recording.shape[1]):
        #     recording[:, j:j + 1] = scaler.transform(recording[:, j:j + 1])

        recording_len = recording.shape[0]
        if recording_len > target_length:
            data[i] = recording[0: target_length]
        else:
            data[i, 0:recording_len] = recording
            mask_zeros_begin = int(float(recording_len) / target_length * mask_length)
            feature_map_mask[i, mask_zeros_begin:] = 0

        current_labels = get_labels(header)
        raw_labels.append(current_labels)
        for label in current_labels:
            if label in classes:
                j = classes.index(label)
                labels[i, j] = 1

    # offline category mapping
    if category_mapping_mat is not None:
        labels = category_mapping.offline_mapping(labels, category_mapping_mat, deepth=4)

    # mapped labels
    if category_mapping_mat is not None:
        category_mapping_without_diag_mat = category_mapping_mat.copy()
        for i in range(len(category_mapping_without_diag_mat)):
            category_mapping_without_diag_mat[i, i] = 0
        mapped_labels = np.matmul(labels, category_mapping_without_diag_mat)
        mapped_labels[mapped_labels > 1] = 1
        unmapped_labels = labels - mapped_labels
    else:
        unmapped_labels = None

    present_label_index = np.amax(labels, axis=0)
    class_mask = np.zeros((num_recordings, len(present_label_index)), dtype=np.float)
    for i in range(num_recordings):
        class_mask[i] = present_label_index
        if unmapped_labels is not None and unmapped_labels[i].any():
            uncovered_labels = category_mapping_without_diag_mat[:, unmapped_labels[i]>0].max(axis=1)
            class_mask[i] -= uncovered_labels

    class_mask[class_mask < 0] = 0

    return data, labels, raw_labels, class_mask, feature_map_mask


def offline_mapping(source_labels, classes, source_to_target_mapping):
    mapped_labels = list()
    for label in source_labels:
        if label in classes:
            mapped_labels.append(label)

        if label in source_to_target_mapping.keys():
            mapped_labels.extend(source_to_target_mapping[label])

    mapped_labels = list(set(mapped_labels))
    return mapped_labels


def train_model(params, model_directory, X, Y, mask, feature_map_mask, domain_id,
                X_test, Y_test, mask_test, feature_map_mask_test, Y_test_labels,
                classes, leads, target_fs, target_length, model_basename, testset_name, epochs=100,
                time_stamp=None, evaluate_only_scored_classes=False, gamma=0., sample_weights=None,
                batch_size=64, b=0.3):

    training_from_scratch = False
    if time_stamp is None:
        training_from_scratch = True
        time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # time_stamp = '20211013-125801'
    # time_stamp = '20211107-150926'
    model_basename = model_basename + '_' + testset_name + '_' + time_stamp + '_gamma' + str(gamma)
    model_directory_sub = os.path.join(model_directory, model_basename)

    if not os.path.exists(model_directory_sub):
        os.mkdir(model_directory_sub)

    cvscores = []

    K.clear_session()

    model = model_structure.ResNet_ClassWiseAttention_MultiScale(**params)
    # model = model_structure.ResNet_ClassWiseAttention_SoftmaxMaximumMapping(**params)
    # model = model_structure.ResNet_ClassWiseAttention_AtomicMapping(**params)
    # model = model_structure.ResNet_Attention_OnlineMapping(**params)
    # model = model_structure.ResNet_Attention(**params)
    # model = model_structure.ResNet_RSC(**params)
    train_frequency = Y.sum(axis=0)/Y.shape[0]
    test_frequency = Y_test.sum(axis=0)/Y_test.shape[0]
    test_train_frequency_ratio = test_frequency / (train_frequency + 1e-10)

    # sample_weights = np.matmul(Y, test_train_frequency_ratio)

    print('test_train_frequency_ratio: \n', test_train_frequency_ratio)

    records_num = len(X)
    class_frequencies = np.sum(Y, axis=0, keepdims=True).astype('float32')
    class_frequencies[class_frequencies == 0] = records_num/2
    # alpha_0 = (records_num / (records_num - class_frequencies)) ** 0.3
    # alpha_1 = (records_num / class_frequencies) ** 0.3

    alpha_0, alpha_1 = np.ones_like(class_frequencies), np.ones_like(class_frequencies)

    alpha_1 *= test_train_frequency_ratio

    alpha_1[alpha_1 > 10] = 10
    alpha_1[alpha_1 < 1] = 1

    print('alpha_0: ', alpha_0)
    print('alpha_1: ', alpha_1)

    alpha_0 = tf.constant(alpha_0, tf.float32)
    alpha_1 = tf.constant(alpha_1, tf.float32)

    # model.compile(loss=[focal_loss_2(gamma=2, alpha=0.9)],
    #               optimizer=adam,
    #               metrics=['binary_accuracy'])
    model_filename = os.path.join(model_directory_sub, 'model')
    temp_best_model = model_filename + "_temp_best.model"
    log_filename = os.path.join(model_directory_sub, 'log.txt')

    # training_records_num = int(0.9 * len(X))
    # X_train = X[0:training_records_num]
    # Y_train = Y[0:training_records_num]
    # mask_train = mask[0:training_records_num]
    # feature_map_mask_train = feature_map_mask[0:training_records_num]
    # domain_id_onehot_train = domain_id_onehot[0:training_records_num]
    #
    # X_eval = X[training_records_num:]
    # Y_eval = Y[training_records_num:]
    # mask_eval = mask[training_records_num:]
    # feature_map_mask_eval = feature_map_mask[training_records_num:]
    # domain_id_onehot_eval = domain_id_onehot[training_records_num:]

    csv_logger = CSVLogger(log_filename)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model_checkpoint = ModelCheckpoint(
        temp_best_model,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1)

    def decay(epoch):
        if epoch < 2:
            lr = 1e-4 * (epoch * 5 + 1)
        else:
            lr = 1e-3 * 0.9**(epoch-2)

        # tf.summary.scalar('learning rate', data=lr, step=epoch)
        return lr

    lr_scheduler = LearningRateScheduler(decay, verbose=1)

    logdir = "logs/scalars-onlinemap/" + time_stamp
    # file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    # file_writer.set_as_default()
    tensorboard_callback = TensorBoard(log_dir=logdir)

    if params['output_sum_mapping']:
        loss = ['binary_crossentropy', category_sum_loss()]
    else:
        loss = ['binary_crossentropy']

    adam = Adam()
    model.compile(
        # loss=[class_correlation_weighted_loss(gamma=0, alpha=0.75, power=1)],
        loss=loss,
        # loss=[focal_loss_2(gamma=2., alpha_0=alpha_0, alpha_1=alpha_1), 'mse'],
        # loss=[focal_loss_2(gamma=gamma, alpha_0=alpha_0, alpha_1=alpha_1)],
        loss_weights=[1, 1],
        optimizer=adam,
        metrics=['binary_accuracy'])

    if params['with_class_mask']:
        inputs_train = [X, mask]
        inputs_test = [X_test, mask_test]
    else:
        inputs_train = X
        inputs_test = X_test

    sum_thres = 2
    if params['output_sum_mapping']:
        output_train = [Y, np.ones_like(Y) * sum_thres]
        output_test = [Y_test, np.ones_like(Y_test) * sum_thres]
    else:
        output_train = [Y]
        output_test = [Y_test]

    train_steps_per_epoch = int(X.shape[0] / batch_size)
    # model.fit_generator(weighted_batch_generator(inputs_train, [Y], sample_weights, batch_size),
    #                     validation_data=(inputs_test, [Y_test]),
    #                     steps_per_epoch=train_steps_per_epoch,
    #                     # validation_split=0.1,
    #                     # validation_data=weighted_batch_generator(X, mask, Y, sample_weights, 64),
    #                     # sample_weight=sample_weights,
    #                     # batch_size=64,
    #                     verbose=2,
    #                     callbacks=[csv_logger, early_stopping, model_checkpoint, lr_scheduler],
    #                     epochs=epochs)

    if training_from_scratch:
        model.fit(inputs_train, output_train,
                  # validation_data=(inputs_test, [Y_test]),
                  validation_split=0.1,
                  # sample_weight=sample_weights,
                  batch_size=64,
                  verbose=2,
                  callbacks=[csv_logger, early_stopping, model_checkpoint, lr_scheduler],
                  epochs=epochs)

    model.load_weights(temp_best_model)

    results = model.evaluate(inputs_test, output_test)
    print('Evaluating results. ')
    print(model.metrics_names)
    print(results)

    if params['output_sum_mapping']:
        Y_test_pred_prob, _ = model.predict(inputs_test)
    else:
        Y_test_pred_prob = model.predict(inputs_test)
    Y_test = np.array(Y_test)

    # maskout minority classes
    # label_frequencies = Y_test.sum(axis=0)
    # label_valid_indexes = np.where(label_frequencies > 50)[0]
    # Y = Y[:, label_valid_indexes]
    # Y_test = Y_test[:, label_valid_indexes]
    # Y_test_pred_prob = Y_test_pred_prob[:, label_valid_indexes]
    # classes = [classes[i] for i in label_valid_indexes]
    # thres = thres[label_valid_indexes]

    if evaluate_only_scored_classes:
        weights_file = 'weights-V2.csv'
        classes_scored, weights = load_weights2(weights_file)
        Y = arrange_labels(Y, classes, classes_scored)
        Y_test = arrange_labels(Y_test, classes, classes_scored)
        Y_test_pred_prob = arrange_labels(Y_test_pred_prob, classes, classes_scored)
        classes_scored = [list(classes_scored[i])[0] for i in range(len(classes_scored))]

        Y_pred, thres = cost_based_categorize_v3(Y_test_pred_prob, Y, Y_test, weights, b=b, lower_limit=0.01)

        scored_log_file = os.path.join(model_directory_sub,f'scorelog_only_scored_b{b}.csv')
        class_wise_scores_file = os.path.join(model_directory_sub, f'class-wise-scores_only_scored_b{b}.csv')
    else:
        classes_scored = classes
        Y_pred, thres = cost_based_categorize_v2(Y_test_pred_prob, Y, Y_test, b=b, lower_limit=0.01)

        scored_log_file = os.path.join(model_directory_sub, f'scorelog_b{b}.csv')
        class_wise_scores_file = os.path.join(model_directory_sub, f'class-wise-scores_b{b}.csv')

    # save model
    save_model(model_filename, classes, leads, target_fs, target_length, model, thres, params)

    # score model
    # Compute F_beta measure and the generalization of the Jaccard index

    _, auroc, auprc, auroc_classes, auprc_classes, accuracy, f_measure, f_beta_measure, challenge_metric, precision, recall = \
        evaluate_model(Y_test, Y_pred, Y_test_pred_prob, classes_scored)

    class_wise_scores = {
        "class codes": classes_scored,
        "class names": get_category_field(classes_scored),
        # "frequency train": list(np.sum(Y, axis=0)),
        "frequency test": list(np.sum(Y_test, axis=0)),
        "auroc_classes": list(auroc_classes),
        "auprc_classes": list(auprc_classes),
        # "thres": list(thres),
        "precision": list(precision),
        "recall": list(recall),
        "F_measure_classes": list(f_beta_measure),
    }
    print('classes number: ', len(classes))
    print('auroc_classes number: ', len(auroc_classes))
    print('precision number: ', len(precision))
    print('F_measure_classes number: ', len(f_beta_measure))
    print(class_wise_scores)
    classwise_scores_df = pd.DataFrame(class_wise_scores)
    classwise_scores_df.to_csv(class_wise_scores_file)

    score = [classes, auroc, auprc, auroc_classes, auprc_classes, accuracy, f_measure, f_beta_measure,
             challenge_metric]
    cvscores.append(score)

    print('----------- Segment classification scores -----------')
    print('auroc:', auroc)
    print('auprc:', auprc)
    print('auroc_classes:', auroc_classes)
    print('auprc_classes:', auprc_classes)
    print('accuracy:', accuracy)
    print('f_measure:', f_measure)
    print('Fbeta_measure:', f_beta_measure)
    print('challenge_metric:', challenge_metric)

    cvscores_head = ['classes', 'auroc', 'auprc', 'auroc_classes', 'auprc_classes', 'accuracy',
                     'f_measure', 'Fbeta_measure', 'challenge_metric']
    with open(scored_log_file, 'w', newline='') as out_f:  # Python 3
        w = csv.writer(out_f, delimiter=',')  # override for tab delimiter
        w.writerow(cvscores_head)
        w.writerows(cvscores)


def compute_threshold(reference, predictions):
    precision, recall, thresholds = precision_recall_curve(reference, predictions)
    # compute f scores
    fscore = (2 * precision * recall) / (precision + recall + 1e-10)
    if np.any(np.isnan(fscore)):
        print('fscore has Nan: ', fscore)
        print('precision: ', precision)
        print('recall: ', recall)
        print('thresholds: ', thresholds)
        thres = thresholds[0]
    else:
        # locate the index of the largest f score
        ix = np.nanargmax(fscore)
        thres = thresholds[ix]
    return thres


################################################################################
#
# File I/O functions
#
################################################################################

# Save your trained models.
def save_model(filename, classes, leads, input_fs, input_len, model, thres, params):
    # Construct a data structure for the model and save it.
    d = {'classes': classes, 'leads': leads, 'input_fs': input_fs, 'input_len': input_len,
         'thres': thres, 'params': params}
    joblib.dump(d, filename + '.sav', protocol=0)
    model.save(filename + '.model')


# Load a trained model. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
def load_model(model_directory, leads):
    filename = os.path.join(model_directory, get_model_filename(leads))
    settings = joblib.load(filename)
    filename = filename[:-4]
    settings['model'] = tf.keras.models.load_model(filename + '.model',
                                                   custom_objects={'focal_loss_fixed': focal_loss_2()})
    return settings


# Define the filename(s) for the trained models. This function is not required. You can change or remove it.
def get_model_filename(leads):
    if leads:
        sorted_leads = sort_leads(leads)
        name = 'model_' + '-'.join(sorted_leads) + '.sav'
    else:
        name = 'model.sav'
    return name

################################################################################
#
# Running trained model functions
#
################################################################################

# Run your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_twelve_lead_model(model, header, recording):
    return run_model(model, header, recording)


# Run your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_six_lead_model(model, header, recording):
    return run_model(model, header, recording)


# Run your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_three_lead_model(model, header, recording):
    return run_model(model, header, recording)


# Run your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_two_lead_model(model, header, recording):
    return run_model(model, header, recording)


# Generic function for running a trained model.
def run_model(model, header, recording, classes_scored=None):
    classes = model['classes']
    leads = model['leads']
    classifier = model['model']
    input_len = model['input_len']
    thres = model['thres']
    target_fs = 250

    if classes_scored is None:
        classes_scored = classes

    # Load data.
    data = np.expand_dims(recording, axis=0)
    pred_mask = np.ones([1, len(classes)], dtype='float32')

    # Predict labels and probabilities.
    probabilities = classifier.predict([data, pred_mask])

    # Arrange prediction
    probabilities = arrange_labels(probabilities, classes, classes_scored)[0]

    labels = (probabilities > thres).astype(np.int)

    return classes_scored, labels, probabilities


def run_model2(model, header, recording):
    classes = model['classes']
    leads = model['leads']
    classifier = model['model']
    input_len = model['input_len']
    thres = model['thres']
    target_fs = 250

    # Load data.
    data = np.expand_dims(recording, axis=0)

    # Predict labels and probabilities.
    probabilities = classifier.predict(data)[0]

    # arrange to scored labels
    weights_file = 'weights-V2.csv'
    classes_scored, weights = load_weights2(weights_file)
    probabilities = arrange_labels(probabilities, classes, classes_scored)

    labels = (probabilities > thres).astype(np.int)

    return classes, labels, probabilities

################################################################################
#
# Other functions
#
################################################################################

# Extract features from the header and recording.
def get_features(header, recording, leads):
    # Extract age.
    age = get_age(header)
    if age is None:
        age = float('nan')

    # Extract sex. Encode as 0 for female, 1 for male, and NaN for other.
    sex = get_sex(header)
    if sex in ('Female', 'female', 'F', 'f'):
        sex = 0
    elif sex in ('Male', 'male', 'M', 'm'):
        sex = 1
    else:
        sex = float('nan')

    # Reorder/reselect leads in recordings.
    available_leads = get_leads(header)
    indices = list()
    for lead in leads:
        i = available_leads.index(lead)
        indices.append(i)
    recording = recording[indices, :]

    # Pre-process recordings.
    adc_gains = get_adcgains(header, leads)
    baselines = get_baselines(header, leads)
    num_leads = len(leads)
    for i in range(num_leads):
        recording[i, :] = (recording[i, :] - baselines[i]) / adc_gains[i]

    # Compute the root mean square of each ECG lead signal.
    rms = np.zeros(num_leads, dtype=np.float32)
    for i in range(num_leads):
        x = recording[i, :]
        rms[i] = np.sqrt(np.sum(x ** 2) / np.size(x))

    return age, sex, rms


def unison_shuffled_copies(array_list):
    p = np.random.permutation(len(array_list[0]))
    shuffled_list = [a[p] for a in array_list]
    return shuffled_list, p


def one_hot_encoding(id_array):
    # get unique values
    ids = np.unique(id_array)
    # construct one-hot array
    onehot_array = np.zeros((len(id_array), len(ids)), dtype='float32')
    for i in range(len(ids)):
        onehot_array[np.equal(id_array[:,0], ids[i]), i] = 1

    return onehot_array


def compute_imbalance_costs(y, lower_limit=0.0):

    y_complement = 1 - y
    zero_frequences = np.sum(y_complement, axis=0)
    one_frequences = np.sum(y, axis=0)

    imbalance_costs = one_frequences / zero_frequences
    imbalance_costs[imbalance_costs < lower_limit] = lower_limit

    return imbalance_costs


def cost_based_categorize(scalar_outputs, labels, b, lower_limit=0.05):
    costs_1_0 = compute_imbalance_costs(labels, lower_limit=lower_limit) ** b
    costs_0_1 = np.ones_like(costs_1_0)

    binary_outputs = np.zeros_like(scalar_outputs)
    thres = np.zeros((scalar_outputs.shape[1],))
    for i in range(scalar_outputs.shape[1]):
        thres[i] = costs_1_0[i] / (costs_1_0[i] + costs_0_1[i])
        binary_outputs[:, i] = (scalar_outputs[:, i] > thres[i]).astype(np.int)

    print('thres: ', thres)

    return binary_outputs, thres


def compute_metric_costs(y, class_weights=None):

    y_normalized = y / np.maximum(np.sum(y, axis=-1, keepdims=True), 1)
    class_correlation = np.matmul(y_normalized, class_weights)
    costs = 1 - class_correlation

    y_complement = 1 - y
    zero_frequences = np.sum(y_complement, axis=0)

    negative_costs = costs * y_complement
    metric_costs = np.sum(negative_costs, axis=0) / zero_frequences

    return metric_costs


def compute_imbalance_costs_v2(y, lower_limit=0.0):

    y_complement = 1 - y
    zero_frequences = np.sum(y_complement, axis=0)
    one_frequences = np.sum(y, axis=0)

    imbalance_costs = one_frequences / zero_frequences
    imbalance_costs[imbalance_costs < lower_limit] = lower_limit

    return imbalance_costs


def cost_based_categorize_v2(scalar_outputs, labels_train, labels_test, b, lower_limit=0.05):
    costs_1_0 = compute_imbalance_costs_v2(labels_train, lower_limit=lower_limit) ** (1 - b)
    # costs_0_1 = np.ones_like(costs_1_0)
    costs_0_1 = np.amax(labels_test, axis=0)

    binary_outputs = np.zeros_like(scalar_outputs)
    thres = np.zeros((scalar_outputs.shape[1],))
    for i in range(scalar_outputs.shape[1]):
        thres[i] = costs_1_0[i] / (costs_1_0[i] + costs_0_1[i])
        binary_outputs[:, i] = (scalar_outputs[:, i] > thres[i]).astype(np.int)

    print('thres: ', thres)

    return binary_outputs, thres


def cost_based_categorize_v3(scalar_outputs, labels_train, labels_test, weights, b, lower_limit=0.05):
    costs_1_0 = compute_metric_costs(labels_train, weights) ** b \
                * compute_imbalance_costs(labels_train,lower_limit=lower_limit) ** (1 - b)
    # costs_1_0 = compute_imbalance_costs_v2(labels_train, lower_limit=lower_limit) ** (1 - b)
    # costs_0_1 = np.ones_like(costs_1_0)
    costs_0_1 = np.amax(labels_test, axis=0)

    binary_outputs = np.zeros_like(scalar_outputs)
    thres = np.zeros((scalar_outputs.shape[1],))
    for i in range(scalar_outputs.shape[1]):
        thres[i] = costs_1_0[i] / (costs_1_0[i] + costs_0_1[i])
        binary_outputs[:, i] = (scalar_outputs[:, i] > thres[i]).astype(np.int)

    print('thres: ', thres)

    return binary_outputs, thres
