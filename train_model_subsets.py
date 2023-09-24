#!/usr/bin/env python

# Do *not* edit this script.

import sys
import os
from team_code import training_code, cross_validation_with_simulated_db_difference, \
    cross_validation_with_simulated_db_difference_2

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    # Parse arguments.
    # if len(sys.argv) != 4:
    #     raise Exception('Include the data and model folders as arguments, e.g., python train_model.py data model.')

    # data_directory = sys.argv[1]
    # model_directory = sys.argv[2]
    # online_mapping = sys.argv[3]

    data_directory = 'E:\\workspaceLY\\CINC2020\\official phase\\datasets\\WFDB_Ningbo'
    model_directory = 'models_subsets_BCE_softmax'

    # Run the training code.
    print('Running training code...')

    # simulate annotation differences among datasets
    masked_categories = [{'59118001': ['713427006'],  # RBBB: CRBBB;
                          '164909002': ['733534002'],  # LBBB: CLBBB
                          '55930002': ['164930006', '429622005']},  # STC: STIAb, STD,
                         {'59118001': ['713426002', '713427006'],  # RBBB: IRBBB, CRBBB
                          '195039008': ['270492004', '195042002'],  # PAB: IAVB, IIAVB
                          '55930002': ['164930006']},  # STC: STIAb
                         {'164909002': ['445211001', '445118002'],  # LBBB: LPFB, LAnFB,
                          '195042002': ['426183003', '54016002'],  # IIAVB: IIAVBII, MoI
                          '55930002': ['429622005'],  # STC: STD
                          '365418004': ['59931005']},  # T wave findings: TInV
                         {'164909002': ['251120003', '733534002'],  # LBBB: ILBBB, CLBBB
                          '233917008': ['195039008'],  # AVB: PAB
                          '195039008': ['270492004', '195042002']},  # PAB: IAVB, IIAVB
                         {'164909002': ['445211001', '445118002', '251120003', '733534002'],
                          # LBBB: LPFB, LAnFB, ILBBB, CLBBB
                          '233917008': ['27885002'],  # AVB: CHB
                          '365418004': ['164934002', '59931005']}]  # T wave findings: TAb, TInV

    cross_validation_with_simulated_db_difference_2(data_directory, model_directory, masked_categories=masked_categories)

    print('Done.')
