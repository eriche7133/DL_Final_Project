import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import _use_shared_memory


class FinalDataset(Dataset):
    def __init__(self, file_names, directory, id_dict):
        self.file_names = file_names
        self.directory = directory
        self.id_dict = id_dict
    
    def __len__(self):
        return(len(self.file_names))
    
    def __getitem__(self, index):
        filename = self.file_names[index]
        filename_list = filename.split('-')
        # print(filename)
        # print(int(filename_list[0]))
        label = self.id_dict[filename_list[0]]
        # label = self.id_dict[int(filename_list[0])]
        data = torch.FloatTensor(np.load(self.directory + '/' + filename))
        data = torch.transpose(data,0,1)
        label_out = torch.LongTensor(1)
        label_out[0] = label
        # print(filename_list[0])
        # print(label)
        # print(data.size())
        # label = torch.LongTensor(label)
        # print(label)
        return (data, label)


def utt_collate_fn(batch):
    batch_size = len(batch)
    # print(batch[0].size())
    feat_dim = batch[0][0].shape[1]
    if _use_shared_memory:
        utt_lengths = torch.LongStorage()._new_shared(batch_size).zero_()
    else:
        utt_lengths = torch.LongTensor(batch_size)
    
    for i in range(batch_size):
        utt_lengths[i] = batch[i][0].shape[0]
    # lengths = torch.LongTensor(np.array(lengths))

    # Sort the feat lengths
    ####
    # (utt_lengths_out, indices) = torch.sort(utt_lengths, descending=True)
    # utt_lengths = utt_lengths[indices]
    (utt_lengths, indices) = torch.sort(utt_lengths, descending=True)

    # Sort the labels according to the lengths of the feats
    
    # labels_sort = labels[indices]
    # label_lengths = torch.IntTensor(label_lengths)
    # lengths_total = torch.sum(label_lengths)
    max_length = utt_lengths[0]
    #### FIND MAX LABEL LENGTH
    if _use_shared_memory:
        utt_stack = torch.FloatStorage()._new_shared(utt_lengths[0] * batch_size * feat_dim).new(utt_lengths[0], batch_size, feat_dim).zero_()
        labels_out = torch.LongStorage()._new_shared(batch_size).zero_()
    else:
        utt_stack = torch.zeros(utt_lengths[0], batch_size, feat_dim).float() # L, B, D, +1 in dimension is to keep the length here
        labels_out = torch.zeros(batch_size).long()
    
    # Create padded stacks of utterances and labels
    # utt_stack = torch.zeros(lengths[0], len(feats), feats[0].shape[1]+1).float() # L, B, D, +1 in dimension is to keep the length here
        
    # utt_stack = torch.zeros(lengths[0], batch_size, feats[0].shape[1]).float() # L, B, D, +1 in dimension is to keep the length here
    # labels_out = torch.zeros(int(lengths_total)).int()
    # label_stack = torch.zeros(torch.max(label_lengths), len(labels_sort), 2).long() # L_label, B, 2, +1 in dimension to keep the length here
    # Modify this part to create
    for i in range(batch_size):
        utt_length = utt_lengths[i]
        utt_stack[:utt_length, i, :] = batch[indices[i]][0]
        # print(batch[indices[i]][1])
        # print(type(labels_out[i]))
        labels_out[i] = int(batch[indices[i]][1]) # +1 because Warp-CTC
    return (utt_stack, utt_lengths, labels_out)


def create_dataloader(file_names, directory, id_dict, args):
    # kwargs = {'pin_memory': True} if args.cuda else {}
    final_dataset = FinalDataset(file_names, directory, id_dict)
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}
    data_loader = DataLoader(
        final_dataset, shuffle=True,
        batch_size=args.batch_size, collate_fn=utt_collate_fn, **kwargs)
    
    # data_loader = UtteranceDataloader(dataset, labels, batch_size=args.batch_size,
      #                                 shuffle=True, **kwargs)
    return data_loader
