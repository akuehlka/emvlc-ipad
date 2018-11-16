# -*- coding: utf-8 -*-

import os
import sys
import itertools
import numpy as np
from glob import glob
from antispoofing.mcnns.datasets.dataset import Dataset
from antispoofing.mcnns.utils import *
import pdb


class NDContactLenses(Dataset):

    def __init__(self, dataset_path, ground_truth_path='', permutation_path='', iris_location='',
                 output_path='./working', file_types=('.png', '.bmp', '.jpg', '.tiff'),
                 operation='crop', max_axis=320,
                 ):

        super(NDContactLenses, self).__init__(dataset_path, output_path, file_types)

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

    def read_permutation_file(self):

        data = np.genfromtxt(self.permutation_path, dtype=np.str, delimiter=',', skip_header=1)

        permuts_index = data[:, 0].astype(np.int)
        permuts_partition = data[:, 1]

        return permuts_index, permuts_partition

    # @profile
    def _build_meta(self, inpath, filetypes):

        img_idx = 0

        all_fnames = []
        all_labels = []
        all_idxs = []

        train_idxs = []
        test_idxs = []

        hash_sensor = []

        folders = [self._list_dirs(inpath, filetypes)]
        # -- flat and sort list of fnames
        folders = itertools.chain.from_iterable(folders)
        folders = sorted(list(folders))

        gt_dict = read_csv_file(self.ground_truth_path, as_dict=True, sequenceid_col=0)
        gt_list = read_csv_file(self.ground_truth_path, as_dict=False)
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

        for i, folder in enumerate(folders):
            fnames = [glob(os.path.join(inpath, folder, '*' + filetype)) for filetype in filetypes]
            fnames = itertools.chain.from_iterable(fnames)
            fnames = sorted(list(fnames))

            for fname in fnames:

                rel_path = os.path.relpath(fname, inpath)
                img_id, ext = os.path.splitext(os.path.basename(rel_path))
                img_id = img_id.split('_')[0]

                line = []
                try:
                    line = gt_dict[img_id]
                except KeyError:
                    pass

                # -- check if the current image is in the ground truth file
                if line:
                    if img_id in sequence_id_train:
                        all_idxs += [img_idx]
                        all_fnames += [fname]
                        train_idxs += [img_idx]
                        hash_sensor += [gt_dict[img_id][self.SENSOR_ID_COL]]

                        # -- Two-class problem: attack vs. real
                        all_labels += [self.ATTEMPTED_ATTACK_CLASS_LABEL if img_id in sequence_id_pos_class else self.GENUINE_CLASS_LABEL]
                        img_idx += 1

                    elif img_id in sequence_id_test:
                        all_idxs += [img_idx]
                        all_fnames += [fname]
                        test_idxs += [img_idx]
                        hash_sensor += [gt_dict[img_id][self.SENSOR_ID_COL]]

                        # -- Two-class problem: attack vs. real
                        all_labels += [self.ATTEMPTED_ATTACK_CLASS_LABEL if img_id in sequence_id_pos_class else self.GENUINE_CLASS_LABEL]
                        img_idx += 1

                    else:
                        pass

        all_fnames = np.array(all_fnames)
        all_labels = np.array(all_labels)
        all_idxs = np.array(all_idxs)
        train_idxs = np.array(train_idxs)
        test_idxs = np.array(test_idxs)

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
                  'train_idxs': train_idxs,
                  'test_idxs': test_idxs,
                  }

        return r_dict

    def _crop_img(self, img, cx, cy, max_axis, padding=0):
        new_height = max_axis
        new_width = max_axis

        cy -= new_height // 2
        cx -= new_width // 2

        if (cy + new_height) > img.shape[0]:
            shift = (cy + new_height) - img.shape[0]
            cy -= shift

        if (cx + new_width) > img.shape[1]:
            shift = (cx + new_width) - img.shape[1]
            cx -= shift

        cy = max(0, cy)
        cx = max(0, cx)

        cx = padding if cx == 0 else cx
        cy = padding if cy == 0 else cy

        cropped_img = img[cy - padding:cy + new_height + padding, cx - padding:cx + new_width + padding, :]

        return cropped_img

    def _get_iris_region(self, fnames):

        origin_imgs, _, _ = load_images(fnames, tmppath=self.hdf5_tmp_path, dsname=type(self).__name__.lower())
        iris_location_dict = read_csv_file(self.iris_location, as_dict=True, sequenceid_col=0)

        imgs = []
        for i, fname in enumerate(fnames):

            key = os.path.splitext(os.path.basename(fname))[0]

            try:
                cx, cy = iris_location_dict[key][1:3]
                cx = int(float(cx))
                cy = int(float(cy))
            except:
                print('Warning: Iris location not found. Cropping in the center of the image', fname)
                cy, cx = origin_imgs[i].shape[0]//2, origin_imgs[i].shape[1]//2

            img = self._crop_img(origin_imgs[i], cx, cy, self.max_axis)
            imgs += [img]

        imgs = np.array(imgs, dtype=np.uint8)

        return imgs

    def get_imgs(self):

        try:
            return self.__imgs
        except AttributeError:
            fnames = self.meta_info['all_fnames']

            if 'segment' in self.operation:
                self.__imgs = self._get_iris_region(fnames)
            else:
                self.__imgs, _, _ = load_images(fnames, tmppath=self.hdf5_tmp_path, dsname=type(self).__name__.lower())

            self.__imgs = np.ascontiguousarray(self.__imgs)

            return self.__imgs

    @property
    def meta_info(self):
        try:
            return self.__meta_info
        except AttributeError:
            self.__meta_info = self._build_meta(self.dataset_path, self.file_types)
            return self.__meta_info

    def meta_info_feats(self, output_path, file_types):
        return self._build_meta(output_path, file_types)
