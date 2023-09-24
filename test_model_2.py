#!/usr/bin/env python

# Do *not* edit this script.
import csv
import pandas as pd
import numpy as np, os, sys

from data_prepare import load_data3
from evaluate_model import evaluate_model, get_category_field, load_weights2, arrange_labels
from team_code import load_model, run_model, twelve_leads, load_data2
from helper_code import *

# Test model.
def test_model(model_directory, data_directory, evaluate_only_scored_classes=False, using_offline_mapping=True):

    # load model
    model_with_settings = load_model(model_directory, None)
    model = model_with_settings['model']
    classes = model_with_settings['classes']
    target_fs = model_with_settings['input_fs']
    target_length = model_with_settings['input_len']
    thres = model_with_settings['thres']
    params = model_with_settings['params']
    mapping_matrix = params['mapping_matrix']

    model.summary()

    # Load test data.
    print('Loading test data ...')
    data, labels, raw_labels, class_mask, feature_map_mask = \
        load_data3(data_directory,
                   target_fs=target_fs,
                   target_length=target_length,
                   mask_length=39,
                   classes=classes, category_mapping_mat=mapping_matrix,
                   leads=twelve_leads,
                   maskout_unmapped=True,
                   using_offline_mapping=using_offline_mapping)

    # Run model for each recording.
    print('Running model...')
    pred_prob = model.predict([data, np.ones_like(class_mask)])
    pred_prob = np.array(pred_prob)

    # maskout minority classes
    label_frequencies = labels.sum(axis=0)
    label_valid_indexes = np.where(label_frequencies > 50)[0]
    pred_prob = pred_prob[:, label_valid_indexes]
    labels = labels[:, label_valid_indexes]
    classes = [classes[i] for i in label_valid_indexes]

    thres = 0.2
    print('thres: ', thres)
    pred_binary = (pred_prob > thres).astype('float32')

    # Scoring
    if evaluate_only_scored_classes:
        weights_file = 'weights-V2.csv'
        classes_scored, weights = load_weights2(weights_file)
        labels = arrange_labels(labels, classes, classes_scored)
        pred_binary = arrange_labels(pred_binary, classes, classes_scored)
        pred_prob = arrange_labels(pred_prob, classes, classes_scored)
        classes = [list(classes_scored[i])[0] for i in range(len(classes_scored))]

        scored_log_file = os.path.join(model_directory, 'test2_scorelog_only_scored.csv')
        class_wise_scores_file = os.path.join(model_directory, 'test2_class-wise-scores_only_scored.csv')
    else:
        scored_log_file = os.path.join(model_directory, 'test2_scorelog.csv')
        class_wise_scores_file = os.path.join(model_directory, 'test2_class-wise-scores.csv')

    _, auroc, auprc, auroc_classes, auprc_classes, accuracy, f_measure, f_beta_measure, challenge_metric, precision, recall = \
        evaluate_model(labels, pred_binary, pred_prob, classes)

    class_wise_scores = {
        "class codes": classes,
        "class names": get_category_field(classes),
        # "frequency train": list(np.sum(Y, axis=0)),
        "frequency test": list(np.sum(labels, axis=0)),
        "auroc_classes": list(auroc_classes),
        "auprc_classes": list(auprc_classes),
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
    classwise_scores_df.to_csv(class_wise_scores_file[:-4] + '_thres' + str(thres) + class_wise_scores_file[-4:])

    cvscores = []
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
    scored_log_file = scored_log_file[:-4] + '_thres' + str(thres) + scored_log_file[-4:]
    with open(scored_log_file, 'w', newline='') as out_f:  # Python 3
        w = csv.writer(out_f, delimiter=',')  # override for tab delimiter
        w.writerow(cvscores_head)
        w.writerows(cvscores)

    print('Done.')

if __name__ == '__main__':

    # model_directory = 'models_G12EC_BCE_upsampling_OffMap\ResNet-ClassWiseAtten_G12EC-label-mask-full-max_20211204-203827_gamma0.0'
    # data_directory = 'E:\\workspaceLY\\CINC2020\\official phase\\datasets\\E\\WFDB'
    # test_model(model_directory, data_directory, evaluate_only_scored_classes=False, using_offline_mapping=True)
    #
    # model_directory = 'models_G12EC_BCE_upsampling_OffMap\ResNet-ClassWiseAtten_G12EC-label-mask-full_20211204-223004_gamma0.0'
    # data_directory = 'E:\\workspaceLY\\CINC2020\\official phase\\datasets\\E\\WFDB'
    # test_model(model_directory, data_directory, evaluate_only_scored_classes=False, using_offline_mapping=True)

    model_directory = 'models_G12EC_BCE_upsampling_OffMap\ResNet-ClassWiseAtten_G12EC-baseline_20211204-150429_gamma0.0'
    data_directory = 'E:\\workspaceLY\\CINC2020\\official phase\\datasets\\E\\WFDB'
    test_model(model_directory, data_directory, evaluate_only_scored_classes=False, using_offline_mapping=True)
    #
    # model_directory = 'models_G12EC_BCE_upsampling_NoOffMap\ResNet-ClassWiseAtten_G12EC-baseline_20211203-142205_gamma0.0'
    # data_directory = 'E:\\workspaceLY\\CINC2020\\official phase\\datasets\\E\\WFDB'
    # test_model(model_directory, data_directory, evaluate_only_scored_classes=False, using_offline_mapping=True)



