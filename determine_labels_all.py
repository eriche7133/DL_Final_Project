import os
from os import listdir
from os.path import isfile, join
import numpy as np
import librosa

def determine_labels_all():
    load_directory = '/home/datasets/Speaker_Recognition/train_wav/VAD_1/mel_hop'
    filenames = [f for f in listdir(load_directory) if isfile(join(load_directory, f))]
    label_list = []
    for counter, filename in enumerate(filenames,1):
        print('{}/{}'.format(counter,len(filenames)))
        filename_list = filename.split('-')
        label = filename_list[0]
        label_list.append(int(label))
    label_list = list(set(label_list))
    np.save('labels_all_10s.npy',label_list)
    labels_all_dict = dict(zip(label_list,list(range(len(label_list)))))
    np.save('labels_all_dict_10s.npy',labels_all_dict)
    return label_list, labels_all_dict
    


def apply_melspectrogram_to_dir(load_directory, save_filename_label, save_filename_mel):
    filenames = [f for f in listdir(load_directory) if isfile(join(load_directory, f))]
    mel_list = []
    label_list = []
    for counter, filename in enumerate(filenames,1):
        print('{}/{}'.format(counter,len(filenames)))
        filename_list = filename.split('-')
        label = filename_list[0]
        # print(label)
        label_list.append(int(label))
        # print(filename)
        mel_list.append(apply_melspectrogram_to_file(load_directory + '/' + filename))
    mel_list = np.array(mel_list)
    label_list = np.array(label_list)
    np.save(save_filename_label, label_list)
    np.save(save_filename_mel, mel_list)


def apply_melspectrogram_to_dir_v2(load_directory, label_directory, mel_directory):
    filenames = [f for f in listdir(load_directory) if isfile(join(load_directory, f))]
    for counter, filename in enumerate(filenames,1):
        print('{}/{}'.format(counter,len(filenames)))
        label_filename = label_directory + '/' + filename[0:-4] + '_label.npy'
        mel_filename = mel_directory + '/' + filename[0:-4] + '_mel.npy'
        if not os.path.isfile(mel_filename):
            mel = apply_melspectrogram_to_file(load_directory + '/' + filename)
            if mel is not None:
                mel = np.array(mel)
                filename_list = filename.split('-')
                label = np.array(filename_list[0])
                np.save(label_filename, label)
                # print(label)
                # print(filename)
                np.save(mel_filename, mel)


def apply_melspectrogram_to_file(filename):
    y, sr = librosa.load(filename)
    if y.shape[0] == 0:
        return None
    else:
        # print(y.shape)
        window_time = .025
        n_fft = sr * window_time
        # print(int(n_fft))
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, n_fft=int(n_fft))
        return spectrogram

def main_fn():
    save_directory = '/home/datasets/Speaker_Recognition/train_wav/mel'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    for aggressiveness in range(4):
        load_directory = '/home/datasets/Speaker_Recognition/train_wav/VAD_' + str(aggressiveness)
        save_filename_label = save_directory + '/train_label_VAD_' + str(aggressiveness) + '.npy'
        save_filename_mel =  save_directory + '/train_mel_VAD_' + str(aggressiveness) + '.npy'
        apply_melspectrogram_to_dir(load_directory, save_filename_label, save_filename_mel)


def main_fn_v2():
    save_directory = '/home/datasets/Speaker_Recognition/train_wav/mel'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    # Only VAD 3 for now
    for aggressiveness in range(3,4):
        load_directory = '/home/datasets/Speaker_Recognition/train_wav/VAD_' + str(aggressiveness)
        label_directory = '/home/datasets/Speaker_Recognition/train_wav/VAD_' + str(aggressiveness) + '/label'
        if not os.path.exists(label_directory):
            os.makedirs(label_directory)
        mel_directory = '/home/datasets/Speaker_Recognition/train_wav/VAD_' + str(aggressiveness) + '/mel'
        if not os.path.exists(mel_directory):
            os.makedirs(mel_directory)
        apply_melspectrogram_to_dir_v2(load_directory, label_directory, mel_directory)

if __name__ == '__main__':
    main_fn_v2()
