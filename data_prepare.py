import math
import wfdb
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

import smooth
from helper_code import find_challenge_files, load_header, load_recording, get_labels


def weighted_batch_generator(X, Y, weights, batch_size):
    num_records = X[0].shape[0]

    while True:
        random_choice = np.random.choice(num_records, batch_size, p=weights)
        X_batch = [X_element[random_choice] for X_element in X]
        Y_batch = [Y_element[random_choice] for Y_element in Y]

        yield X_batch, Y_batch


def load_data2(data_directory, target_fs=250, target_length=5000, mask_length=39, classes=None,
              category_mapping_mat=None, leads=None, maskout_unmapped=True, using_offline_mapping=True):
    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)
    num_classes = len(classes)

    if not num_recordings:
        raise Exception('No data was provided.')

    data = []
    labels = []  # One-hot encoding of classes
    feature_map_mask = []
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

        current_labels = get_labels(header)
        label_vector = np.zeros((num_classes,), dtype='float32')
        for label in current_labels:
            if label in classes:
                j = classes.index(label)
                label_vector[j] = 1
        if np.amax(label_vector) == 0:
            continue
        else:
            labels.append(label_vector)
            raw_labels.append(current_labels)

        recording_len = recording.shape[0]
        if recording_len > target_length:
            data.append(recording[0: target_length])
            feature_map_mask.append(np.ones((target_length,)))
        else:
            data.append(recording)
            mask_zeros_begin = int(float(recording_len) / target_length * mask_length)
            mask = np.ones((target_length,))
            mask[mask_zeros_begin:] = 0
            feature_map_mask.append(mask)

    num_recordings = len(data)

    data = pad_sequences(data, maxlen=target_length, padding='post', truncating='post', dtype='float32')
    labels = np.array(labels, dtype='float32')
    feature_map_mask = np.array(feature_map_mask, dtype='float32')

    # offline category mapping
    if using_offline_mapping and category_mapping_mat is not None:
        # labels = category_mapping.offline_mapping(labels, category_mapping_mat, deepth=4)
        labels = np.matmul(labels, category_mapping_mat)
        labels[labels > 0] = 1

    # mapped labels
    if category_mapping_mat is not None:
        class_mask = compute_class_mask(category_mapping_mat, labels, maskout_unmapped)
    else:
        class_mask = None

    return data, labels, raw_labels, class_mask, feature_map_mask


def load_data3(data_directory, target_fs=250, target_length=5000, mask_length=39, classes=None,
               category_mapping_mat=None, leads=None, maskout_unmapped=True, using_offline_mapping=True):

    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)
    num_classes = len(classes)

    if not num_recordings:
        raise Exception('No data was provided.')

    data = []
    labels = []  # One-hot encoding of classes
    feature_map_mask = []
    raw_labels = []

    for i in range(num_recordings):
        # print('    {}/{}...'.format(i + 1, num_recordings))

        # load ECG data
        recording, temporal_mask = load_record(header_files[i], target_fs, target_length, mask_length)

        if math.isnan(recording.min()):
            continue

        # load labels
        label_vector, current_labels = load_labels(header_files[i], classes)

        if np.amax(label_vector) == 0:
            continue
        else:
            data.append(recording)
            labels.append(label_vector)
            raw_labels.append(current_labels)
            feature_map_mask.append(temporal_mask)

    data = pad_sequences(data, maxlen=target_length, padding='post', truncating='post', dtype='float32')
    labels = np.array(labels, dtype='float32')
    feature_map_mask = np.array(feature_map_mask, dtype='float32')

    print('raw data min: ', data.min())
    print('raw data max: ', data.max())
    # print('raw data min mean: ', np.mean(data_mins))
    # print('raw data min std: ', np.std(data_mins))
    # print('raw data max mean: ', np.mean(data_maxs))
    # print('raw data max std: ', np.std(data_maxs))

    # offline category mapping
    if using_offline_mapping and category_mapping_mat is not None:
        # labels = category_mapping.offline_mapping(labels, category_mapping_mat, deepth=4)
        labels = np.matmul(labels, category_mapping_mat)
        labels[labels > 0] = 1

    # mapped labels
    if category_mapping_mat is not None:
        class_mask = compute_class_mask(category_mapping_mat, labels, maskout_unmapped)
    else:
        class_mask = None

    return data, labels, raw_labels, class_mask, feature_map_mask


def compute_class_mask(category_mapping_mat, labels, maskout_unmapped):

    category_mapping_without_diag_mat = category_mapping_mat.copy()
    for i in range(len(category_mapping_without_diag_mat)):
        category_mapping_without_diag_mat[i, i] = 0
    mapped_labels = np.matmul(labels, category_mapping_without_diag_mat)
    mapped_labels[mapped_labels > 1] = 1
    unmapped_labels = labels - mapped_labels

    present_label_index = np.amax(labels, axis=0)
    num_recordings = len(labels)
    class_mask = np.zeros((num_recordings, len(present_label_index)), dtype=np.float)
    for i in range(num_recordings):
        class_mask[i] = present_label_index
        if maskout_unmapped and unmapped_labels is not None and unmapped_labels[i].max() > 0:
            try:
                uncovered_labels = category_mapping_without_diag_mat[:, unmapped_labels[i]>0].max(axis=1)
            except:
                print('')
                print(unmapped_labels[i])
                print('')

            class_mask[i] -= uncovered_labels

    class_mask[class_mask < 0] = 0
    return class_mask


def bandpass_filter(data, lowcut, highcut, signal_freq, filter_order):
    """
    Method responsible for creating and applying Butterworth filter.
    :param deque data: raw data
    :param float lowcut: filter lowcut frequency value
    :param float highcut: filter highcut frequency value
    :param int signal_freq: signal frequency in samples per second (Hz)
    :param int filter_order: filter order
    :return array: filtered data
    """
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    data_filtered = np.zeros_like(data, dtype=np.float32)
    for i in range(data.shape[-1]):
        b, a = butter(filter_order, [low, high], btype="bandpass", output='ba')
        data_filtered[:, i] = filtfilt(b, a, data[:, i])

    return data_filtered


def load_record(header_file, target_fs, target_length, mask_length):
    # Load header and recording.
    recording, fields = wfdb.rdsamp(header_file[:-4])

    # get sampling frequency of the file
    fs = int(fields['fs'])

    if fs != target_fs:
        step = round(fs / target_fs)
        signal_length = recording.shape[0]
        recording = recording[0:signal_length:step, :]

    # remove baseline wander
    for j in range(recording.shape[1]):
        smoothed_signal = smooth.smooth(recording[:, j], window_len=target_fs, window='flat')
        recording[:, j] = recording[:, j] - smoothed_signal

    # band pass filtering
    recording = bandpass_filter(recording, 0.1, 50, target_fs, 1)

    # normalization
    scaler = StandardScaler()
    scaler.fit(recording[:, 7:8])  # fit the signal in Lead V2
    for j in range(recording.shape[1]):
        recording[:, j:j + 1] = scaler.transform(recording[:, j:j + 1])

    # padding or truncate the recording to the target length
    recording_len = recording.shape[0]
    channels = recording.shape[1]
    if recording_len > target_length:
        recording = recording[0: target_length]
        temporal_mask = np.ones((mask_length,))
    else:
        recording_pad = np.zeros((target_length, channels), dtype='float32')
        recording_pad[0:recording_len, :] = recording
        recording = recording_pad

        mask_zeros_begin = int(float(recording_len) / target_length * mask_length)
        temporal_mask = np.ones((mask_length,))
        temporal_mask[mask_zeros_begin:] = 0

    return recording, temporal_mask


def load_labels(header_file, classes):
    header = load_header(header_file)
    num_classes = len(classes)

    labels = get_labels(header)
    label_vector = np.zeros((num_classes,), dtype='float32')
    for label in labels:
        if label in classes:
            j = classes.index(label)
            label_vector[j] = 1

    return label_vector, labels

