# -*- coding: utf-8 -*-

import os
import itertools
import numpy as np
from glob import glob
from antispoofing.mcnns.datasets.dataset import Dataset
from antispoofing.mcnns.utils import *


class LivDetIris17_Warsaw(Dataset):
    def __init__(self, dataset_path, ground_truth_path='', permutation_path='', iris_location='',
                 output_path='./working', file_types=('.png', '.bmp', '.jpg', '.tiff'),
                 operation='crop', max_axis=320,
                 ):

        super(LivDetIris17_Warsaw, self).__init__(dataset_path, output_path, iris_location, file_types, operation,
                                                  max_axis)

        self.ground_truth_path = ground_truth_path
        self.LIV_DET_TRAIN = os.path.join(self.ground_truth_path, 'train.txt')
        self.LIV_DET_TEST = os.path.join(self.ground_truth_path, 'test.txt')
        self.LIV_DET_UNKNOWN_TEST = os.path.join(self.ground_truth_path, 'unknown_test.txt')
        self.verbose = True

    def _build_meta(self, inpath, filetypes):

        all_fnames = []
        all_labels = []

        liv_det_train_data, _ = read_csv_file(self.LIV_DET_TRAIN, sequenceid_col=0, delimiter=',',
                                              remove_header=True)
        liv_det_test_data, _ = read_csv_file(self.LIV_DET_TEST, sequenceid_col=0, delimiter=',',
                                             remove_header=True)
        liv_det_unknown_test_data, _ = read_csv_file(self.LIV_DET_UNKNOWN_TEST,
                                                     sequenceid_col=0, delimiter=',',
                                                     remove_header=True)

        # train images
        train_fnames = [os.path.join(inpath, f[0]) for f in liv_det_train_data]
        all_fnames += train_fnames
        all_labels += [int(l) for l in liv_det_train_data[:,1]]
        train_idxs = np.arange(liv_det_train_data.shape[0])

        # test images
        test_fnames = [os.path.join(inpath, f[0]) for f in liv_det_test_data]
        all_fnames += test_fnames
        all_labels += [int(l) for l in liv_det_test_data[:,1]]
        ixstart = max(train_idxs)+1
        ixstop = ixstart + liv_det_test_data.shape[0]
        test_idxs = np.arange(ixstart, ixstop)

        # unknown test images
        unk_test_fnames = [os.path.join(inpath, f[0]) for f in liv_det_unknown_test_data]
        all_fnames += unk_test_fnames
        all_labels += [int(l) for l in liv_det_unknown_test_data[:,1]]
        ixstart = max(test_idxs)+1
        ixstop = ixstart + liv_det_unknown_test_data.shape[0]
        unknown_test_idxs = np.arange(ixstart, ixstop)

        all_fnames = np.array(all_fnames)
        all_labels = np.array(all_labels)
        all_idxs = np.arange(all_fnames.shape[0])
        train_idxs = np.array(train_idxs)
        test_idxs = np.array(test_idxs)
        unknown_test_idxs = np.array(unknown_test_idxs)

        all_pos_idxs = np.where(all_labels == self.POS_LABEL)
        all_neg_idxs = np.where(all_labels == self.NEG_LABEL)

        r_dict = {'all_fnames': all_fnames,
                  'all_labels': all_labels,
                  'all_idxs': all_idxs,
                  'all_pos_idxs': all_pos_idxs,
                  'all_neg_idxs': all_neg_idxs,
                  'train_idxs': train_idxs,
                  'test_idxs': {'test': test_idxs,
                                'unknown_test': unknown_test_idxs,
                                },
                  }

        if self.verbose:
            self.info(r_dict)

        return r_dict

    def protocol_eval(self, fold=0, n_fold=5, train_size=0.5):

        # -- loading the training data and its labels
        all_fnames = self.meta_info['all_fnames']
        all_labels = self.meta_info['all_labels']
        train_idxs = self.meta_info['train_idxs']
        test_idxs = self.meta_info['test_idxs']

        all_data = self.get_imgs(all_fnames)

        train_set = {'data': all_data[train_idxs],
                     'labels': all_labels[train_idxs],
                     'idxs': train_idxs}

        test_set = {}
        for test_id in test_idxs:
            if test_idxs[test_id].size:
                test_set[test_id] = {'data': all_data[test_idxs[test_id]],
                                     'labels': all_labels[test_idxs[test_id]],
                                     'idxs': test_idxs[test_id],
                                     }

        return {'train_set': train_set, 'test_set': test_set}
