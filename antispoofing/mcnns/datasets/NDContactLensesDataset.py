# -*- coding: utf-8 -*-

import os
import sys
import itertools
import numpy as np
from glob import glob
from antispoofing.mcnns.datasets.dataset import Dataset
from antispoofing.mcnns.utils import *
import pdb


class NDContactLensesDataset_15(Dataset):

    def __init__(self, dataset_path, ground_truth_path='', permutation_path='', iris_location='',
                 output_path='./working', file_types=('.png', '.bmp', '.jpg', '.tiff'),
                 operation='crop', max_axis=320,
                 ):

        super(NDContactLensesDataset_15, self).__init__(dataset_path, output_path, iris_location, file_types, operation, max_axis)

        self.ground_truth_path = ground_truth_path
        self.permutation_path = permutation_path
        self.iris_location = iris_location

        self.SEQUENCE_ID_COL = 0
        self.SENSOR_ID_COL = 3
        self.CONTACTS_COL = 4

        # -- classes
        self.ATTEMPTED_ATTACK_CLASS_LABEL = 1
        self.GENUINE_CLASS_LABEL = 0

        self.operation = operation
        self.max_axis = max_axis

        self.verbose = True

        self.__imgs = []

        # loads a map with the path to each image in the dataset
        self.file_mapping = self.__read_file_mapping()

        # public attribute to be used when bsif extraction is called
        # self.imgs = []

    def read_permutation_file(self):

        data = np.genfromtxt(self.permutation_path, dtype=np.str, delimiter=',', skip_header=1)

        permuts_index = data[:, 0].astype(np.int)
        permuts_partition = data[:, 1]

        return permuts_index, permuts_partition

    def __read_file_mapping(self):
        flist = glob(self.dataset_path + "/*.tiff")

        pathmap = {}
        for f in flist:
            sid = f.split('.')[0].split('/')[-1]
            pathmap[sid] = f

        return pathmap


    # @profile
    def _build_meta(self, inpath, filetypes):

        img_idx = 0

        all_fnames = []
        all_labels = []
        all_idxs = []

        train_idxs = []
        test_idxs = []

        hash_sensor = []

        folders = [self.list_dirs(inpath, filetypes)]
        # -- flat and sort list of fnames
        folders = itertools.chain.from_iterable(folders)
        folders = sorted(list(folders))

        gt_list, gt_dict = read_csv_file(self.ground_truth_path, sequenceid_col=0)
        permuts_index, permuts_partition = self.read_permutation_file()

        # -- samples of the Positive class (Attempted Attack Images)
        with_contact_texture_idxs = np.where(gt_list[:, self.CONTACTS_COL] == 'T')[0].reshape(1, -1)
        with_contact_lenses_idxs = with_contact_texture_idxs

        # -- samples of the Negative class (Genuine Images)
        without_contact_lenses_idxs = np.where(gt_list[:, self.CONTACTS_COL] == 'N')[0].reshape(1, -1)

        sequence_id_pos_class = gt_list[with_contact_lenses_idxs, self.SEQUENCE_ID_COL]
        sequence_id_neg_class = gt_list[without_contact_lenses_idxs, self.SEQUENCE_ID_COL]

        if self.verbose:
            print('with_contact_texture_idxs', with_contact_texture_idxs.shape)
            print('without_contact_lenses_idxs', without_contact_lenses_idxs.shape)
            print('sequence_id_pos_class', sequence_id_pos_class.shape)
            print('sequence_id_neg_class', sequence_id_neg_class.shape)
            sys.stdout.flush()

        idxs = np.where(permuts_partition == '0.8')[0]
        sequence_id_train = gt_list[permuts_index[idxs], self.SEQUENCE_ID_COL]

        idxs = np.where(permuts_partition == '0.2')[0]
        sequence_id_test = gt_list[permuts_index[idxs], self.SEQUENCE_ID_COL]

        subject_id = []
        i = 0
        # read the file path for all images, and keep track of indexes and labels,
        # using the file mapping dictionary
        for sid in [r[self.SEQUENCE_ID_COL] for r in gt_list]:
            try:
                all_fnames += [self.file_mapping[sid]]
                hash_sensor += [gt_list[gt_dict[sid]][self.SENSOR_ID_COL]]
                subject_id += [sid]
                all_idxs += [i]
                all_labels += [self.ATTEMPTED_ATTACK_CLASS_LABEL if sid in sequence_id_pos_class
                               else self.GENUINE_CLASS_LABEL]
                if sid in sequence_id_train:
                    train_idxs += [i]
                if sid in sequence_id_test:
                    test_idxs += [i]
                i += 1
            except KeyError:
                print("Key Error for ", sid)
                continue

        # save train/test indexes on properties to be used later
        self.train_idxs = train_idxs
        self.test_idxs = test_idxs

        all_fnames = np.array(all_fnames)
        all_labels = np.array(all_labels)
        all_idxs = np.array(all_idxs)
        subject_id = np.array(subject_id)
        train_idxs = np.array(train_idxs)
        test_idxs = np.array(test_idxs)

        all_pos_idxs = np.where(all_labels[all_idxs] == self.POS_LABEL)[0]
        all_neg_idxs = np.where(all_labels[all_idxs] == self.NEG_LABEL)[0]

        if self.verbose:
            print("-- all_fnames:", all_fnames.shape)
            print("-- all_labels:", all_labels.shape)
            print("-- all_idxs:", all_idxs.shape)
            print("-- train_idxs:", train_idxs.shape)
            print("-- test_idxs:", test_idxs.shape)
            sys.stdout.flush()

        r_dict = {'all_fnames': all_fnames,
                  'all_labels': all_labels,
                  'all_idxs': all_idxs,
                  'all_pos_idxs': all_pos_idxs,
                  'all_neg_idxs': all_neg_idxs,
                  'subject_id': subject_id,
                  'train_idxs': train_idxs,
                  'test_idxs': {'test': test_idxs,
                                },
                  }

        return r_dict

    def meta_info_feats(self, output_path, file_types):
        return self._build_meta(output_path, file_types)

    def protocol_eval(self, fold=0, n_fold=5, train_size=0.5):
        # -- loading the training data and its labels
        all_fnames = self.meta_info['all_fnames']
        all_labels = self.meta_info['all_labels']
        subject_id = self.meta_info['subject_id']
        all_data = self.get_imgs(all_fnames)

        # split the data according to the permutation that was loaded
        train_set = {'data':all_data[self.train_idxs],
                     'labels':all_labels[self.train_idxs],
                     'idxs':self.train_idxs}
        test_set = {'test':{'data':all_data[self.test_idxs],
                            'labels':all_labels[self.test_idxs],
                            'idxs':self.test_idxs}
                    }

        return {'train_set': train_set, 'test_set': test_set}
