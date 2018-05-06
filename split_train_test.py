import argparse
import sys
import collections
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import final_dataset as fd
import torch


def filenames_hop():
    load_directory = '/home/datasets/Speaker_Recognition/train_wav/VAD_1/mel_hop'
    file_names = [f for f in listdir(load_directory) if isfile(join(load_directory, f))]
    file_names_train = file_names
    np.save('file_names_train_notest_hop.npy',file_names_train)


def filenames_hop_subset(subset_num):
    load_directory = '/home/datasets/Speaker_Recognition/train_wav/VAD_1/mel_hop'
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
    np.save('labels_all_10s_subset_' + str(subset_num) + '.npy',label_list)
    labels_all_dict = dict(zip(label_list,list(range(len(label_list)))))
    np.save('labels_all_dict_10s_subset_' + str(subset_num) + '.npy',labels_all_dict)
    np.save('file_names_train_notest_hop_subset_' + str(subset_num) + '.npy',file_names2)


def split_train_notest(load_directory):
    file_names = [f for f in listdir(load_directory) if isfile(join(load_directory, f))]
    file_names_train = file_names
    np.save('file_names_train_notest.npy',file_names_train)


def split_train_test(load_directory):
    file_names = [f for f in listdir(load_directory) if isfile(join(load_directory, f))]
    num_files = len(file_names)
    train_size = int(num_files*.8)
    file_perm = np.random.permutation(num_files)
    file_names_train = file_names[:train_size]
    file_names_test = file_names[train_size:]
    np.save('file_names_train.npy',file_names_train)
    np.save('file_names_test.npy',file_names_test)


def test_data_loader():
    load_directory = '/home/datasets/Speaker_Recognition/train_wav/VAD_1/mel'
    parser = create_parser()
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    labels_all_dict = np.load('labels_all_dict.npy')
    file_names_train = np.load('file_names_train.npy')
    file_names_test = np.load('file_names_test.npy')
    train_dataloader = fd.create_dataloader(file_names_train, load_directory, labels_all_dict.item(), args)
    test_dataloader =  fd.create_dataloader(file_names_test, load_directory, labels_all_dict.item(), args)
    
    it = 0;
    for x,y,z in train_dataloader:
        
        print('x.shape: {}'.format(x.shape))
        print('y.shape: {}'.format(y.shape))
        print('z.shape: {}'.format(z.shape))
        
        it = it+1
        print(it)

def create_parser():
    parser = argparse.ArgumentParser(description='HW3P3')
    parser.add_argument('--save_directory', type=str, default='output/inferno/train4',
                        help='output directory')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--filename', type=str, default='test_output2.csv',
                        help='test results filename (default: posttraining_output_test.csv)')
    parser.add_argument('--load_model', action='store_true', default=True,
                        help='Load Saved Data')
    parser.add_argument('--load_directory', type=str, default='output/inferno/train4',
                        help='output directory')
    parser.add_argument('--small_train', action='store_true', default=False,
                        help='Train on Dev only')
    parser.add_argument('--test_only', action='store_true', default=False,
                        help='Test Only (loaded a model and want to apply it to  test data only)')
    parser.add_argument('--input_dim', type=int, default=40, metavar='N',
                        help='Number of dimensions of data')
    parser.add_argument('--hidden_dim', type=int, default=256, metavar='N',
                        help='Number of hidden dimensions')
    parser.add_argument('--key_dim', type=int, default=128, metavar='N',
                        help='Number of dimensions in key and value output')
    parser.add_argument('--num_workers', type=int, default=1, metavar='N',
                        help='Number of hidden dimensions')
    parser.add_argument('--random_search_iters', type=int, default=1000, metavar='N',
                        help='Number of hidden dimensions')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--data_directory', type=str, default='dataset',
                        help='data directory')
    parser.add_argument('--words_not_chars', action='store_true', default=True,
                        help='Use words as the dictionary, not characters')
    return parser

