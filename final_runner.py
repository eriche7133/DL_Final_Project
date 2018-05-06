import argparse
import sys
import collections

import os

import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import functional as F

from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger

# import Levenshtein as L
import final_dataset as fd
import final_module as fm
# from CTC_Loss_Inferno import *


def to_tensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array).float()


def to_variable(tensor):
    # Tensor -> Variable (on GPU if possible)
    if torch.cuda.is_available():
        # Tensor -> GPU Tensor
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)


class cust_CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input, target):
        loss = self.loss_fn(input, target)
        print_values = False
        if print_values:
            # predictions = torch.max(input, 1)[1]
            # print(input)
            # print(predictions)
            # print(target)
            print(loss)
        return loss


def test_model(model, args):
    test_feats = load_test_feats(args.data_directory)
    test_dataloader = create_test_dataloader(test_feats, args)
    print('Loaded Test Data')
    loss_fn = SeqCrossEntropyLoss(args)
    test_dataloader_flag = False
    if test_dataloader_flag:
        it = 0;
        for v,w,x,y,z in test_dataloader:
            
            print('v.shape: {}'.format(v.shape))
            print('w.shape: {}'.format(w.shape))
            print('x.shape: {}'.format(x.shape))
            print('y.shape: {}'.format(y.shape))
            print('z.shape: {}'.format(z.shape))
            
            it = it+1
            print(it)

    filename = args.filename
    model.eval()
    # final_out = list(range(test_feats.shape[0]))
    final_out = np.array(range(test_feats.shape[0]),dtype=object)
    if args.words_not_chars:
        reverse_converter = load_reverse_converter(args.data_directory)
    else:
        reverse_converter = load_reverse_converter_char(args.data_directory)

    for i, sample in enumerate(test_dataloader):
        sample_feat = sample[0]
        sample_utt_len = sample[1]
        sample_label_len = sample[2]
        sample_label = sample[3]
        sample_target = sample[4]
        if args.cuda:
            sample_feat = to_variable(sample_feat)
            sample_utt_len = to_variable(sample_utt_len)
            sample_label_len = to_variable(sample_label_len)
            sample_label = to_variable(sample_label)
        sample_in = (sample_feat, sample_utt_len, sample_label_len, sample_label)
        model.generator_length = args.generator_length
        (logits, label_lengths) = model(*sample_in)
        # logits is length x batch x dim
        
        model.generator_length = 0
        losses = np.zeros(args.random_search_iters)
        for t in range(args.random_search_iters):
            label = logits[t]
            sample_label_len = torch.LongTensor(1)
            # print(label)
            # print(int(label.index(0)))
            # label_len = label.index(0)
            # print(label)
            label_len = len(label)
            if label_len == 1:
                losses[t] = -1e10
            else:
                sample_label_len[0] = label_len
                sample_label = torch.zeros(label_len, 1).long()
                sample_target = torch.zeros(label_len, 1).long()
                for t2 in range(label_len):
                    if t2 == 0:
                        sample_label[0] = 0
                    else:
                        sample_label[t2] = label[t2-1].data
                    sample_target[t2] = label[t2].data
                # sample_label[0] = 0
                # print(label)
                # sample_label[1:] = torch.LongTensor(label[1:])
                # sample_target = torch.LongTensor(label)
                sample_label_len = to_variable(sample_label_len)
                sample_label = to_variable(sample_label)
                sample_target = to_variable(sample_target)
                sample_in = (sample_feat, sample_utt_len, sample_label_len, sample_label)
                predicted = model(*sample_in)
                losses[t] = loss_fn(predicted, sample_target)
        t_final = np.argmax(losses)
        # From here, convert output of NN to a list of integer labels (removing 0 and 1)
        int_list = []
        for t in range(len(logits[t_final])):
            int_list.append(int(logits[t_final][t].data))
        # int_list = logits[t_final]
        
        # Remove start of utterance characters (0)
        int_list = [x for x in int_list if x != 0]
        # Remove end of utterance characters (1)
        # int_list = [x for x in int_list if x != 1]
        temp_str = ''
        str_list = []
        for int_el in int_list:
            str_list.append(reverse_converter[int_el])
        temp_str = ' '.join(str_list)
        final_out[i] = temp_str
        # final_out[i] = smoothed_logits
        print('Test Model Iteration : {}/{}'.format(i, test_feats.shape[0]))
    # np.save(filename,final_out)
    csvWriteOutput(final_out,filename)


def load_model(args):
    trainer = Trainer()
    trainer.load(from_directory=args.load_directory, best=False)
    # trainer.load(from_directory=args.load_directory, best=False)
    trainer.set_max_num_epochs(args.epochs + trainer.epoch_count)
    model = trainer.model
    trainer.build_logger(TensorboardLogger(log_scalars_every=(1, 'iteration'),
                                        log_images_every='never'),
                      log_directory=args.save_directory)
    trainer.save_to_directory(args.save_directory)
    return (model, trainer)


def train_model(args):
    model = fm.model_fn(args)
    print('Done initializing model')
    # labels_all_dict = np.load('labels_all_dict.npy')
    labels_all_dict = np.load('labels_all_dict_10s_subset_500.npy')
    if args.no_val:
        file_names_train = np.load('file_names_train_notest_hop_subset_500.npy')
        # file_names_train = np.load('file_names_train_notest.npy')
        train_dataloader = fd.create_dataloader(file_names_train, args.data_directory, labels_all_dict.item(), args)
    else:
        file_names_train = np.load('file_names_train.npy')
        file_names_test = np.load('file_names_test.npy')
        train_dataloader = fd.create_dataloader(file_names_train, args.data_directory, labels_all_dict.item(), args)
        dev_dataloader =  fd.create_dataloader(file_names_test, args.data_directory, labels_all_dict.item(), args)
    

    # train_dataloader = fd.create_dataloader(train_feats, train_labels, args)
    # dev_dataloader = fd.create_dataloader(dev_feats, dev_labels, args)

    test_dataloader_flag = False
    if test_dataloader_flag:
        it = 0;
        for x,y,z in train_dataloader:
            
            print('x.shape: {}'.format(x.shape))
            print('y.shape: {}'.format(y.shape))
            print('z.shape: {}'.format(z.shape))
            print('x: {}'.format(x))
            print('y: {}'.format(y))
            print('z: {}'.format(z))
            
            it = it+1
            print(it)

    print('Done Creating Data Loaders')

    if args.load_model:
        (model, trainer) = load_model(args)
        trainer \
            .bind_loader('train', train_dataloader, num_inputs=2, num_targets=1)
        if not args.no_val:
            trainer.bind_loader('validate', dev_dataloader, num_inputs=2, num_targets=1)
    else:
        # Build trainer
        loss_fn = cust_CrossEntropyLoss()
        trainer = Trainer(model) \
            .build_criterion(loss_fn) \
            .build_metric('CategoricalError') \
            .build_optimizer(torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)) \
            .save_every((1, 'epochs')) \
            .save_to_directory(args.save_directory) \
            .set_max_num_epochs(args.epochs) \
            .build_logger(TensorboardLogger(log_scalars_every=(1, 'iteration'),
                                            log_images_every='never'),
                          log_directory=args.save_directory)
        if not args.no_val:
            trainer \
                .save_at_best_validation_score(True) \
                .validate_every((1, 'epochs'))
        trainer \
            .bind_loader('train', train_dataloader, num_inputs=2, num_targets=1)
        if not args.no_val:
            trainer.bind_loader('validate', dev_dataloader, num_inputs=2, num_targets=1)

            # .build_optimizer('Adam') \
            # .save_at_best_validation_score(True) \
            # .build_metric(LevenshteinMetric) \
            # .evaluate_metric_every('300 iterations') \
            

    if args.cuda:
        print('Switch trainer to CUDA')
        trainer.cuda()
    
    trainer.fit()
    print('Done Fitting Trainer')
    return model


def create_parser():
    parser = argparse.ArgumentParser(description='HW3P3')
    parser.add_argument('--save_directory', type=str, default='output/inferno/train1_hop',
                        help='output directory')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--load_model', action='store_true', default=False,
                        help='Load Saved Data')
    parser.add_argument('--load_directory', type=str, default='output/inferno/train10',
                        help='output directory')
    parser.add_argument('--test_only', action='store_true', default=False,
                        help='Test Only (loaded a model and want to apply it to  test data only)')
    parser.add_argument('--input_dim', type=int, default=40, metavar='N',
                        help='Number of dimensions of data')
    parser.add_argument('--hidden_dim', type=int, default=256, metavar='N',
                        help='Number of hidden dimensions')
    parser.add_argument('--speaker_num', type=int, default=500, metavar='N',
                        help='Number of dimensions in key and value output')
    parser.add_argument('--num_workers', type=int, default=1, metavar='N',
                        help='Number of hidden dimensions')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--data_directory', type=str, default='/home/datasets/Speaker_Recognition/train_wav/VAD_1/mel_hop',
                        help='data directory')
    parser.add_argument('--encoder', type=str, default='CNN1D',
                        help='encoder type')
    parser.add_argument('--decoder', type=str, default='MLP',
                        help='decoder type')
    parser.add_argument('--pooling', type=str, default='mean',
                        help='pooling type')
    parser.add_argument('--no_val', action='store_true', default=True,
                        help='disables validation')
    return parser
    

def main(argv):
    parser = create_parser()
    args = parser.parse_args()
    args.load_directory += '/' + args.encoder + '/' + args.decoder + '/' + args.pooling
    args.save_directory += '/' + args.encoder + '/' + args.decoder + '/' + args.pooling
    if not os.path.exists(args.save_directory):
        os.makedirs(args.save_directory)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    model = train_model(args)
    # if not args.test_only:
        # model = train_model(args)
    # else:
        # (model, _) = load_model(args)
    # test_model(model, args)


if __name__ == '__main__':
    main(sys.argv[1:])
