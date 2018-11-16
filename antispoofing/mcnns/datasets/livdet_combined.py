# -*- coding: utf-8 -*-

import os
import itertools
import numpy as np
from glob import glob
import re
from antispoofing.mcnns.datasets.dataset import Dataset
from antispoofing.mcnns.utils import *


class LivDetCombined(Dataset):
    '''
    This class models a combined set of all Livdet 2017 Datasets
    '''

    def __init__(self,
                 dataset_path,
                 ground_truth_path='',
                 permutation_path='',
                 iris_location='',
                 output_path='./working',
                 file_types=('.png', '.bmp', '.jpg', '.tiff'),
                 operation='crop',
                 max_axis=320,
                 augmentation=False):

        super(LivDetCombined, self).__init__(
            dataset_path,
            output_path,
            iris_location,
            file_types,
            operation,
            max_axis,
            augmentation=augmentation)

        self.ground_truth_path = ground_truth_path

        self.CL_TRAIN = os.path.join(self.ground_truth_path, 'cl_train.csv')
        self.CL_TEST = os.path.join(self.ground_truth_path, 'cl_test.csv')
        self.CL_UNK = os.path.join(self.ground_truth_path, 'cl_unknown.csv')

        self.II_TRAIN = os.path.join(self.ground_truth_path, 'ii_train.csv')
        self.II_UNK = os.path.join(self.ground_truth_path, 'ii_test.csv')

        self.ND_TRAIN = os.path.join(self.ground_truth_path, 'nd_train.csv')
        self.ND_TEST = os.path.join(self.ground_truth_path, 'nd_test.csv')
        self.ND_UNK = os.path.join(self.ground_truth_path, 'nd_unknown.csv')

        self.WW_TRAIN = os.path.join(self.ground_truth_path, 'ww_train.csv')
        self.WW_TEST = os.path.join(self.ground_truth_path, 'ww_test.csv')
        self.WW_UNK = os.path.join(self.ground_truth_path, 'ww_unknown.csv')
        
        self.verbose = True

        # these will be loaded with the metadata
        self.iris_location_list = []
        self.iris_location_hash = {}

    def _load_csv_metadata(self):
        
        cltrain_data, cltrain_hash = read_csv_file(
            self.CL_TRAIN,
            sequenceid_col=0,
            delimiter=',',
            remove_header=True
        )
        cltest_data, cltest_hash = read_csv_file(
            self.CL_TEST,
            sequenceid_col=0,
            delimiter=',',
            remove_header=True
        )
        clunk_data, clunk_hash = read_csv_file(
            self.CL_UNK,
            sequenceid_col=0,
            delimiter=',',
            remove_header=True
        )

        iitrain_data, iitrain_hash = read_csv_file(
            self.II_TRAIN,
            sequenceid_col=0,
            delimiter=',',
            remove_header=True
        )
        iiunk_data, iiunk_hash = read_csv_file(
            self.II_UNK,
            sequenceid_col=0,
            delimiter=',',
            remove_header=True
        )

        ndtrain_data, ndtrain_hash = read_csv_file(
            self.ND_TRAIN,
            sequenceid_col=0,
            delimiter=',',
            remove_header=True
        )
        ndtest_data, ndtest_hash = read_csv_file(
            self.ND_TEST,
            sequenceid_col=0,
            delimiter=',',
            remove_header=True
        )
        ndunk_data, ndunk_hash = read_csv_file(
            self.ND_UNK,
            sequenceid_col=0,
            delimiter=',',
            remove_header=True
        )

        wwtrain_data, wwtrain_hash = read_csv_file(
            self.WW_TRAIN,
            sequenceid_col=0,
            delimiter=',',
            remove_header=True
        )
        wwtest_data, wwtest_hash = read_csv_file(
            self.WW_TEST,
            sequenceid_col=0,
            delimiter=',',
            remove_header=True
        )
        wwunk_data, wwunk_hash = read_csv_file(
            self.WW_UNK,
            sequenceid_col=0,
            delimiter=',',
            remove_header=True
        )

        # join all metadata
        train_data = np.vstack((cltrain_data, iitrain_data, ndtrain_data, wwtrain_data))
        test_data = np.vstack((cltest_data, ndtest_data, wwtest_data))
        unk_data = np.vstack((clunk_data, iiunk_data, ndunk_data, wwunk_data))

        return train_data, test_data, unk_data


    def _build_meta(self, inpath, filetypes):

        all_fnames = []
        all_labels = []

        train_data, test_data, unk_data = self._load_csv_metadata()

        # if self.augmentation:
        #     train_data_tmp = train_data.copy().tolist()
        #
        #     # replicate the training set for adding augmentation effects
        #     data_aug = np.copy(train_data)
        #
        #     for i, item in enumerate(data_aug):
        #         # add replicated data to the original list
        #
        #         itemname = '_B.'.join(item[0].split('.'))  # blur
        #         train_data_tmp.append([itemname, item[1], item[2], item[3]])
        #
        #         itemname = '_I.'.join(item[0].split('.'))  # illumination
        #         train_data_tmp.append([itemname, item[1], item[2], item[3]])
        #
        #         itemname = '_E.'.join(item[0].split('.'))  # edge
        #         train_data_tmp.append([itemname, item[1], item[2], item[3]])
        #
        #     # replace the original list and map with the augmented ones
        #     train_data = np.array(train_data_tmp)

        # train images
        train_fnames = [f[0] for f in train_data]
        all_fnames += train_fnames
        all_labels += [int(l) for l in train_data[:, 1]]
        train_idxs = np.arange(train_data.shape[0])

        # test images
        test_fnames = [f[0] for f in test_data]
        all_fnames += test_fnames
        all_labels += [int(l) for l in test_data[:, 1]]
        ixstart = max(train_idxs) + 1
        ixstop = ixstart + test_data.shape[0]
        test_idxs = np.arange(ixstart, ixstop)

        # unknown images
        unknown_fnames = [f[0] for f in unk_data]
        all_fnames += unknown_fnames
        all_labels += [int(l) for l in unk_data[:, 1]]
        ixstart = max(test_idxs) + 1
        ixstop = ixstart + unk_data.shape[0]
        unknown_idxs = np.arange(ixstart, ixstop)

        all_fnames = np.array(all_fnames)
        all_labels = np.array(all_labels)
        all_idxs = np.arange(all_fnames.shape[0])

        # here we split train into train and validation
        train_idxs_tmp = np.array(train_idxs)
        s = train_idxs_tmp.shape[0]
        np.random.seed(7)
        tix = np.random.choice(s, int(s * 0.8), replace=False)
        vix = np.array(list(set(np.arange(s)) - set(tix)))
        train_idxs = train_idxs_tmp[tix]
        val_idxs = train_idxs_tmp[vix]

        test_idxs = np.array(test_idxs)
        unknown_idxs = np.array(unknown_idxs)

        all_pos_idxs = np.where(all_labels == self.POS_LABEL)[0]
        all_neg_idxs = np.where(all_labels == self.NEG_LABEL)[0]

        # load the iris locations
        # we need them to estimate size when resizing the cropped images
        self.iris_location_list, self.iris_location_hash = read_csv_file(
            self.iris_location, sequenceid_col=0)

        r_dict = {
            'all_fnames': all_fnames,
            'all_labels': all_labels,
            'all_idxs': all_idxs,
            'all_pos_idxs': all_pos_idxs,
            'all_neg_idxs': all_neg_idxs,
            'train_idxs': train_idxs,
            'val_idxs': val_idxs,
            'test_idxs': {
                'test': test_idxs,
                'unknown_test': unknown_idxs,
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
        val_idxs = self.meta_info['val_idxs']
        test_idxs = self.meta_info['test_idxs']

        all_data = self.get_imgs(all_fnames)

        # load the entire dataset to memory, to avoid
        # "IndexError: Selection lists cannot have repeated values"
        data_tmp = np.zeros(all_data.shape, dtype=np.float32)
        try:
            for r in np.arange(all_data.shape[0]):
                data_tmp[r] = all_data[r]
        except ValueError:
            for r in np.arange(all_data.shape[0]):
                data_tmp[r, :, :, :] = all_data[r, :, :, :]
        all_data = data_tmp

        #TODO: find a better solution for this

        # this is the default case
        test_data = {}
        try:
            train_data = all_data[train_idxs]
            val_data = all_data[val_idxs]
            for test_id in test_idxs:
                test_data[test_id] = all_data[test_idxs[test_id]]

        # this is for handling the slicing problem that appears when all_data is a pytable object
        except ValueError:
            train_data = all_data[train_idxs, :, :, :]
            val_data = all_data[val_idxs, :, :, :]
            for test_id in test_idxs:
                test_data['test_id'] = all_data[test_idxs[test_id], :, :, :]

        train_set = {
            'data': train_data,
            'labels': all_labels[train_idxs],
            'idxs': train_idxs
        }

        val_set = {
            'data': val_data,
            'labels': all_labels[val_idxs],
            'idxs': val_idxs
        }

        test_set = {}
        for test_id in test_idxs:
            if test_idxs[test_id].size:
                test_set[test_id] = {
                    'data': test_data[test_id],
                    'labels': all_labels[test_idxs[test_id]],
                    'idxs': test_idxs[test_id],
                }

        return {
            'train_set': train_set,
            'val_set': val_set,
            'test_set': test_set
        }

    # override the ancestor's method
    def _get_iris_region(self, fnames, seqidcol=3):

        origin_imgs, self.iris_location_list, self.iris_location_hash = load_images(
            fnames,
            tmppath=self.hdf5_tmp_path,
            dsname=type(self).__name__.lower(),
            segmentation_data=(self.iris_location_list,
                               self.iris_location_hash))
        error_count = 0

        imgs = []
        for i, fname in enumerate(fnames):

            # for this dataset, we have to use the entire path as the key
            key = re.sub('_[B|I|E]$', '', os.path.splitext(fname)[0])

            try:
                cx, cy = self.iris_location_list[self.iris_location_hash[key]][
                    -3:-1]
                cx = int(float(cx))
                cy = int(float(cy))

            except:
                error_count += 1
                print(
                    'Warning: Iris location not found. Cropping in the center of the image',
                    fname)
                cy, cx = origin_imgs[i].shape[0] // 2, origin_imgs[i].shape[
                    1] // 2

            img = self._crop_img(origin_imgs[i], cx, cy, self.max_axis)
            imgs += [img]

        print("Total of {} iris locations not found.".format(error_count))
        imgs = np.array(imgs, dtype=np.uint8)

        return imgs
