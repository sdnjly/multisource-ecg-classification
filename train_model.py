#!/usr/bin/env python

# Do *not* edit this script.

import sys
import os
from team_code import training_code

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    # Parse arguments.
    # if len(sys.argv) != 4:
    #     raise Exception('Include the data and model folders as arguments, e.g., python train_model.py data model.')

    # data_directory = sys.argv[1]
    # model_directory = sys.argv[2]
    # online_mapping = sys.argv[3]

    data_directory = 'E:\\workspaceLY\\CINC2020\\official phase\\datasets'
    model_directory = 'models_ChapmanShaoxing_BCE_NoOffMap_ablations'

    train_paths = [
        os.path.join(data_directory, "CPSC/Training_WFDB"),
        os.path.join(data_directory, "CPSC2/Training_2"),
        os.path.join(data_directory, "E/WFDB"),
        os.path.join(data_directory, "PTB/WFDB"),
        os.path.join(data_directory, "PTB-XL/WFDB"),
        os.path.join(data_directory, "WFDB_Ningbo")
    ]

    test_path = os.path.join(data_directory, "WFDB_ChapmanShaoxing")
    test_set_name = 'ChapmanShaoxing'

    # Run the training code.
    print('Running training code...')

    maskout_unmapped = True
    using_multi_scale_features = True
    using_dense_SE = True
    training_code(data_directory, train_paths, test_path, test_set_name, model_directory,
                  use_offlinemapping=False,
                  hirarchical_based_mask=maskout_unmapped,
                  online_mapping_type='max',
                  evaluate_only_scored_classes=False,
                  using_class_wise_atten=True,
                  using_global_max_pooling=False,
                  using_multi_scale_features=using_multi_scale_features,
                  output_sum_mapping=False,
                  dense_regularization_type='l2',
                  ecg_poolinglayers=6,
                  using_dense_SE=using_dense_SE,
                  b=0.3)

    # maskout_unmapped = True
    # using_multi_scale_features = False
    # using_dense_SE = True
    # training_code(data_directory, train_paths, test_path, test_set_name+"_SE", model_directory,
    #               use_offlinemapping=True,
    #               hirarchical_based_mask=maskout_unmapped,
    #               online_mapping_type='max',
    #               evaluate_only_scored_classes=True,
    #               using_class_wise_atten=True,
    #               using_global_max_pooling=False,
    #               using_multi_scale_features=using_multi_scale_features,
    #               output_sum_mapping=False,
    #               dense_regularization_type='l2',
    #               ecg_poolinglayers=6,
    #               using_dense_SE=using_dense_SE,
    #               b=0.3)
    #
    # maskout_unmapped = True
    # using_multi_scale_features = True
    # using_dense_SE = False
    # training_code(data_directory, train_paths, test_path, test_set_name + "_multiscale", model_directory,
    #               use_offlinemapping=False,
    #               hirarchical_based_mask=maskout_unmapped,
    #               online_mapping_type='max',
    #               evaluate_only_scored_classes=True,
    #               using_class_wise_atten=True,
    #               using_global_max_pooling=False,
    #               using_multi_scale_features=using_multi_scale_features,
    #               output_sum_mapping=False,
    #               dense_regularization_type='l2',
    #               ecg_poolinglayers=6,
    #               using_dense_SE=using_dense_SE,
    #               b=0.3)

    # maskout_unmapped = True
    # using_multi_scale_features = False
    # using_dense_SE = False
    # training_code(data_directory, train_paths, test_path, test_set_name, model_directory,
    #               use_offlinemapping=False,
    #               hirarchical_based_mask=maskout_unmapped,
    #               online_mapping_type='max',
    #               evaluate_only_scored_classes=True,
    #               using_class_wise_atten=True,
    #               using_global_max_pooling=False,
    #               using_multi_scale_features=using_multi_scale_features,
    #               output_sum_mapping=False,
    #               dense_regularization_type='l2',
    #               ecg_poolinglayers=6,
    #               using_dense_SE=using_dense_SE,
    #               b=0.3)


    print('Done.')
