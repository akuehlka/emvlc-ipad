# -*- coding: utf-8 -*-

import os
import itertools
import numpy as np
from glob import glob
import re
from antispoofing.mcnns.datasets.dataset import Dataset
from antispoofing.mcnns.utils import *
from antispoofing.mcnns.utils.misc import to_vgg_shape


class LivDetIIITD(Dataset):
    '''
    This class is derived from LivDetIris17_Warsaw, but it has a validation split in the training set
    This validation split is used to estimate the predictors accuracy before applying classification
    to known_test
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
                 augmentation=False,
                 transform_vgg=False):

        super(LivDetIIITD, self).__init__(
            dataset_path,
            output_path,
            iris_location,
            file_types,
            operation,
            max_axis,
            augmentation=augmentation,
            transform_vgg=transform_vgg)

        self.ground_truth_path = ground_truth_path
        self.LIV_DET_TRAIN = os.path.join(self.ground_truth_path,
                                          'trainset.csv')
        self.LIV_DET_TEST = os.path.join(self.ground_truth_path, 'testset.csv')
        self.verbose = True

        # these will be loaded with the metadata
        self.iris_location_list = []
        self.iris_location_hash = {}

    def _build_meta(self, inpath, filetypes):

        all_fnames = []
        all_labels = []

        liv_det_train_data, _ = read_csv_file(
            self.LIV_DET_TRAIN,
            sequenceid_col=0,
            delimiter=',',
            remove_header=True)
        liv_det_test_data, _ = read_csv_file(
            self.LIV_DET_TEST,
            sequenceid_col=0,
            delimiter=',',
            remove_header=True)

        # if self.augmentation:
        #     train_data_tmp = liv_det_train_data.copy().tolist()
        #
        #     # replicate the training set for adding augmentation effects
        #     data_aug = np.copy(liv_det_train_data)
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
        #     liv_det_train_data = np.array(train_data_tmp)

        # train images
        train_fnames = [os.path.join(inpath, f[1]) for f in liv_det_train_data]
        all_fnames += train_fnames
        all_labels += [int(l) for l in liv_det_train_data[:, 2]]
        train_idxs = np.arange(liv_det_train_data.shape[0])

        # test images
        test_fnames = [os.path.join(inpath, f[1]) for f in liv_det_test_data]
        all_fnames += test_fnames
        all_labels += [int(l) for l in liv_det_test_data[:, 2]]
        ixstart = max(train_idxs) + 1
        ixstop = ixstart + liv_det_test_data.shape[0]
        test_idxs = np.arange(ixstart, ixstop)

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
                               self.iris_location_hash),
            transform_vgg=self.transform_vgg
            )
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

            if self.transform_vgg:
                img = to_vgg_shape(img)

            imgs += [img]

        print("Total of {} iris locations not found.".format(error_count))
        imgs = np.array(imgs, dtype=np.uint8)

        return imgs
