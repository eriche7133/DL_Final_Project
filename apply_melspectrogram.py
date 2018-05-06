import os
from os import listdir
from os.path import isfile, join
import numpy as np
import librosa

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


def apply_melspectrogram_to_dir_v2(load_directory, mel_directory):
    filenames = [f for f in listdir(load_directory) if isfile(join(load_directory, f))]
    for counter, filename in enumerate(filenames,1):
        print('{}/{}'.format(counter,len(filenames)))
        mel_filename = mel_directory + '/' + filename[0:-4] + '_mel.npy'
        if not os.path.isfile(mel_filename):
            mel = apply_melspectrogram_to_file(load_directory + '/' + filename)
            if mel is not None:
                mel = np.array(mel)
                # print(label)
                # print(filename)
                np.save(mel_filename, mel)


def apply_melspectrogram_to_dir_v3(load_directory, mel_directory, subset_num):
    file_names = [f for f in listdir(load_directory) if isfile(join(load_directory, f))]
    label_list = []
    file_names2 = []
    for counter, filename in enumerate(file_names,1):
        filename_list = filename.split('-')
        label = filename_list[0]
        if label in label_list:
            file_names2.append(filename)
        else:
            if len(label_list) < subset_num:
                label_list.append(label)
                file_names2.append(filename)
    print(len(label_list))
    for counter, filename in enumerate(file_names2,1):
        print('{}/{}'.format(counter,len(file_names2)))
        mel_filename = mel_directory + '/' + filename[0:-4] + '_mel'
        if not os.path.isfile(mel_filename):
            mel = apply_melspectrogram_to_file(load_directory + '/' + filename)
            if mel is not None:
                mel = np.array(mel)
                mel_len = mel.shape[1]
                num_files = mel_len//1000
                ind = 0
                for i in range(num_files-1):
                    ind_len = 1000
                    mel_temp = mel[:,ind:ind+ind_len]
                    # print(mel_temp.shape)
                    np.save(mel_filename + '_' + str(i) + '.npy', mel_temp)
                    ind += ind_len
                # print(label)
                # print(filename)
                # np.save(mel_filename, mel)


def apply_melspectrogram_to_file(filename):
    y, sr = librosa.load(filename)
    if y.shape[0] == 0:
        return None
    else:
        # print(y.shape)
        window_time = .025
        hop_time = .01
        n_fft = sr * window_time
        hop_len = sr*hop_time
        # print(int(n_fft))
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, n_fft=int(n_fft), hop_length = int(hop_len))
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
    # Only VAD 3 for now
    load_directory = '/home/datasets/Speaker_Recognition/train_wav/VAD_1'
    mel_directory = '/home/datasets/Speaker_Recognition/train_wav/VAD_1' + '/mel_hop'
    if not os.path.exists(mel_directory):
        os.makedirs(mel_directory)
    apply_melspectrogram_to_dir_v2(load_directory, mel_directory)


def main_fn_v3():
    # Only VAD 3 for now
    load_directory = '/home/datasets/Speaker_Recognition/train_wav/VAD_1'
    mel_directory = '/home/datasets/Speaker_Recognition/train_wav/VAD_1' + '/mel_hop'
    if not os.path.exists(mel_directory):
        os.makedirs(mel_directory)
    apply_melspectrogram_to_dir_v3(load_directory, mel_directory,500)
if __name__ == '__main__':
    main_fn_v3()
