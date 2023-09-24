from copy import deepcopy

import numpy as np

def get_label_mapping():
    label_mapping = {'426783006': ['427393009', '426177001', '427084000'], # Sinus rhythm: SA, SB, STach
                      '426627000': ['426177001'],   # Brady: SB
                      # '17366009': ['164889003', '164890007', '713422000'], # Atrial arrhythmia: AF, AFL, ATach
                      '164889003': ['195080001', '426749004', '282825002', '314208002'], # AF: AFAFL, CAF, PAF, RAF
                      '164890007': ['195080001'],   # AFL: AFAFL
                      '3424008':  ['427084000', '713422000', '426648003', '164895002'],  # Tach: STach, ATach, JTach, VTach
                      '713422000': ['426761007'],   # ATach: SVT
                      '426761007': ['713422000', '67198005'],   # SVT: ATach, PSVT
                      '426648003': ['251166008', '233897008'],  # JTach: AVNRT, AVRT
                      '164895002': ['425856008', '111288001'],  # VTach: PVT, VFL
                      '60423000': ['65778007','5609005'], # SND: SAB, SARR
                      '29320008': ['426995002', '251164006', '426648003'],   # AVJR: JE, JPC, JTach
                      '284470004': ['63593006', '251170000', '251173003'],  # PAC: SVPB, BPAC, AB
                      '63593006': ['284470004'],    # SVPB: PAC
                      '427172004': ['164884008', '17338001', '251182009', '251180001', '11157007'], # PVC:
                                                                            # VEB, VPB, VPVC, VTrig, VBig
                      '164884008': ['427172004'],   # VEB: PVC
                      '17338001': ['427172004'],    # VPB: PVC
                      '6374002': ['164909002', '59118001'], # BBB: LBBB, RBBB
                      '164909002': ['445211001', '445118002', '251120003', '733534002'],    # LBBB: LPFB, LAnFB, ILBBB, CLBBB
                      '59118001': ['713427006', '713426002'],  # RBBB: CRBBB, IRBBB
                      '233917008': ['195039008', '27885002'],   # AVB: PAB, CHB
                      '195039008': ['270492004', '195042002'],  # PAB: IAVB, IIAVB
                      '195042002': ['426183003', '54016002'],   # IIAVB: IIAVBII, MoI
                      '49260003': ['61277005', '13640000', '11157007', '75532003', '81898007'], # IR: AIVR, FB, VBig, VEsB, VEsR
                      '195126007': ['446813000', '446358003'],  # AH: LAH, RAH
                      '67741000119109': ['446813000'], # LAE: LAH
                      '266249003': ['164873001', '89792004'],   # VH: LVH, RVH
                      '164873001': ['55827005'],    # LVH: LVHV
                      '55827005': ['164873001'],    # LVHV: LVH
                      '253339007': ['67751000119106'], # RAAb: RAHV
                      '67751000119106': ['446358003'],   # RAHV: RAH
                      '365418004': ['164934002', '251259000', '59931005'], # T wave findings: TAb, HTV, TInV
                      '55930002': ['164930006', '428750005', '429622005', '164931005'], # STC: STIAb, NSSTTA, STD, STE
                      '164861001': ['426434006', '425419005', '425623009', '413444003', '413844008'],    # MIs:
                                                                            # AnMIs, IIs, LIs, AMIs, CMI
                      '164865005': ['164867002', '57054005'],   # MI: OldMI, AMI
                      '57054005': ['54329005'],     # AMI: AnMI
                      }
    return label_mapping


def get_label_mapping2():
    label_mapping = {'426783006': ['427393009', '426177001', '427084000'], # Sinus rhythm: SA, SB, STach
                     '6374002': ['164909002', '59118001'], # BBB: LBBB, RBBB
                     '164909002': ['445211001', '445118002', '251120003', '733534002'],    # LBBB: LPFB, LAnFB, ILBBB, CLBBB
                     '59118001': ['713427006', '713426002'],  # RBBB: CRBBB, IRBBB
                     '233917008': ['195039008', '27885002'],  # AVB: PAB, CHB
                     '195039008': ['270492004', '195042002'],  # PAB: IAVB, IIAVB
                     '195042002': ['426183003', '54016002'],  # IIAVB: IIAVBII, MoI
                     '55930002': ['164930006', '429622005'],  # STC: STIAb, STD
                     '365418004': ['164934002', '59931005'],  # T wave findings: TAb, TInV
                      }
    return label_mapping


def get_label_mapping_PTBXL():
    label_mapping = {'426783006': ['427393009', '426177001', '427084000'], # Sinus rhythm: SA, SB, STach
                     '6374002': ['164909002', '59118001'], # BBB: LBBB, RBBB
                     '164909002': ['445211001', '445118002', '251120003', '733534002'],    # LBBB: LPFB, LAnFB, ILBBB, CLBBB
                     '59118001': ['713427006', '713426002'],  # RBBB: CRBBB, IRBBB
                     '233917008': ['195039008', '27885002'],  # AVB: PAB, CHB
                     '195039008': ['270492004', '195042002'],  # PAB: IAVB, IIAVB
                     '195042002': ['426183003', '54016002'],  # IIAVB: IIAVBII, MoI
                     '55930002': ['164930006', '428750005', '429622005', '164931005'], # STC: STIAb, NSSTTA, STD, STE
                     '365418004': ['164934002', '59931005'],  # T wave findings: TAb, TInV
                     '63593006': ['284470004'],  # SVPB: PAC
                     '164884008': ['427172004'],  # VEB: PVC
                     '17338001': ['427172004'],  # VPB: PVC
                     '427172004': ['164884008', '17338001', '251182009', '251180001', '11157007'],  # PVC:
                                                                    # VEB, VPB, VTrig, VBig
                     '266249003': ['164873001', '89792004'],  # VH: LVH, RVH
                     '164873001': ['55827005'],  # LVH: LVHV
                     '55827005': ['164873001'],  # LVHV: LVH
                     '164865005': ['164867002', '57054005'],  # MI: OldMI, AMI
                     '57054005': ['54329005'],  # AMI: AnMI
                     '164861001': ['426434006', '425419005', '425623009', '413444003', '413844008'],  # MIs:
                                            # AnMIs, IIs, LIs, AMIs, CMI
                     # '17366009': ['164889003', '164890007', '195080001', '713422000'],
                                            # Atrial arrhythmia: AF, AFL, AFAFL, ATach

                      }
    return label_mapping


def get_mapping_mat(classes, label_map=None):
    n = len(classes)
    if label_map is None:
        label_map = get_label_mapping()
    mapping_mat = np.zeros((n, n), dtype='float32')
    for i in range(n):
        mapping_mat[i, i] = 1
        if classes[i] in label_map:
            for c in label_map[classes[i]]:
                if c in classes:
                    mapping_mat[classes.index(c), i] = 1

    return mapping_mat


def get_mapping_mat_of_atomic_types(mapping_mat):
    atomic_type_ids = (np.sum(mapping_mat, axis=0) == 1)
    return mapping_mat[atomic_type_ids]


def complete_mapping_matrix(mapping_mat, deepth=1):
    for i in range(deepth):
        mapping_mat = np.matmul(mapping_mat, mapping_mat)

    mapping_mat[mapping_mat > 0] = 1

    return mapping_mat


def offline_mapping(label_onehot_mat, mapping_mat, deepth=1):
    for i in range(deepth):
        label_onehot_mat = np.matmul(label_onehot_mat, mapping_mat)

    label_onehot_mat[label_onehot_mat > 0] = 1

    return label_onehot_mat


def modify_label_level(label_vectors, label_masks, high_to_low_mapping_dict, classes):
    mapping_mat = get_mapping_mat(classes, high_to_low_mapping_dict)
    label_masks_modified = deepcopy(label_masks)
    # remove self-to-self mapping of lower labels
    for high_label in high_to_low_mapping_dict:
        for low_label in high_to_low_mapping_dict[high_label]:
            low_label_id = classes.index(low_label)
            mapping_mat[low_label_id, low_label_id] = 0
            label_masks_modified[:, low_label_id] = 0

    # modify the label vectors with offline mapping
    modified_labels = offline_mapping(label_vectors, mapping_mat, deepth=1)

    return modified_labels, label_masks_modified



