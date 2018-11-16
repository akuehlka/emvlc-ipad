# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import re
from abc import ABCMeta
from abc import abstractmethod
from antispoofing.mcnns.utils import *
from antispoofing.mcnns.utils.misc import to_vgg_shape
from antispoofing.mcnns.features import Extraction


class Dataset(metaclass=ABCMeta):
    def __init__(self, dataset_path, output_path, iris_location, file_types, operation, max_axis, augmentation=False,
                 transform_vgg=False):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.file_types = file_types

        self.iris_location = iris_location
        self.operation = operation
        self.max_axis = max_axis

        # -- classes
        self.POS_LABEL = 1
        self.NEG_LABEL = 0

        # temp directory for cache data
        self.hdf5_tmp_path = '/scratch365/akuehlka/mcnns_tmp'

        self._imgs = []

        # defines if the instance will do augmentation
        self.augmentation = augmentation

        # defines if images will be transformed to "fit" VGG
        self.transform_vgg = transform_vgg

    def prune_train_dataset(self, all_labels, train_idxs):
        # -- prune samples if necessary to have equal sized splits
        neg_idxs = [idx for idx in train_idxs if all_labels[idx] == self.NEG_LABEL]
        pos_idxs = [idx for idx in train_idxs if all_labels[idx] == self.POS_LABEL]
        n_samples = min(len(neg_idxs), len(pos_idxs))

        rstate = np.random.RandomState(7)
        rand_idxs_neg = rstate.permutation(neg_idxs)
        rand_idxs_pos = rstate.permutation(pos_idxs)

        neg_idxs = rand_idxs_neg[:n_samples]
        pos_idxs = rand_idxs_pos[:n_samples]
        train_idxs = np.concatenate((pos_idxs, neg_idxs))

        return train_idxs

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

    def _get_iris_region(self, fnames, seqidcol=3):

        origin_imgs, _, _ = load_images(fnames, tmppath=self.hdf5_tmp_path, 
                                        dsname=type(self).__name__.lower(),
                                        transform_vgg=self.transform_vgg)
        iris_location_list, iris_location_hash = read_csv_file(self.iris_location, sequenceid_col=seqidcol)
        error_count = 0

        imgs = []
        for i, fname in enumerate(fnames):

            key = re.sub('_[B|I|E]$','',os.path.splitext(os.path.basename(fname))[0])

            try:
                cx, cy = iris_location_list[iris_location_hash[key]][-3:-1]
                cx = int(float(cx))
                cy = int(float(cy))
            except:
                error_count += 1
                print('Warning: Iris location not found. Cropping in the center of the image', fname)
                cy, cx = origin_imgs[i].shape[0] // 2, origin_imgs[i].shape[1] // 2

            img = self._crop_img(origin_imgs[i], cx, cy, self.max_axis)

            if self.transform_vgg:
                img = to_vgg_shape(img)

            imgs += [img]

        print("Total of {} iris locations not found.".format(error_count))
        imgs = np.array(imgs, dtype=np.uint8)

        return imgs

    def get_imgs(self, input_fnames):

        if len(self._imgs) > 0:
            return self._imgs
        else:
            fnames = input_fnames

            seqidcol = 0
            if self.__class__.__name__ == 'NDCLD15' or \
                            self.__class__.__name__ == 'NDContactLenses' or \
                            self.__class__.__name__ == 'NDSpoofingPreClassification':
                seqidcol = 3

            if 'crop' in self.operation:
                self._imgs = self._get_iris_region(fnames, seqidcol=seqidcol)
            # TODO: this is not ideal - "segment" operation was missing here, when actually "crop" was doing its job
            elif 'segment' in self.operation:
                self._imgs = self._get_iris_region(fnames, seqidcol=seqidcol)
            else:
                self._imgs, _, _ = load_images(fnames, tmppath=self.hdf5_tmp_path, 
                                               dsname=type(self).__name__.lower(),
                                               transform_vgg=self.transform_vgg)

            return self._imgs

    def check(self, labels_hash, liv_det_train_hash, liv_det_train_data):
        res = []
        base_names = []
        lenses_type = []
        base_names_liv_det = []
        for key in liv_det_train_hash.keys():
            found = 1
            try:
                labels_hash[key]
            except KeyError:
                print('Key not found in labels_hash[key]', key)
                sys.stdout.flush()
                found = 0

            try:
                liv_det_train_data[liv_det_train_hash[key]]
            except KeyError:
                print('Key not found in liv_det_train_data[liv_det_train_hash[key]]', key)
                sys.stdout.flush()
                found = 0

            if found:
                res += [int(liv_det_train_data[liv_det_train_hash[key]][1]) == int(labels_hash[key][1])]
                base_names += [labels_hash[key][0]]
                base_names_liv_det += [liv_det_train_data[liv_det_train_hash[key]][0]]
                lenses_type += [labels_hash[key][2]]

        res = np.array(res).reshape((-1, 1))
        base_names = np.array(base_names).reshape((-1, 1))
        base_names_liv_det = np.array(base_names_liv_det).reshape((-1, 1))
        lenses_type = np.array(lenses_type).reshape((-1, 1))

        return res, base_names, base_names_liv_det, lenses_type

    def info(self, meta_info):

        print('-*- Dataset Info -*-')
        print('-- all_labels:', meta_info['all_labels'].shape)
        print('-- train_idxs:', meta_info['train_idxs'].shape)
        print('   - pos:', np.where(meta_info['all_labels'][meta_info['train_idxs']] == self.POS_LABEL)[0].shape)
        print('   - neg:', np.where(meta_info['all_labels'][meta_info['train_idxs']] == self.NEG_LABEL)[0].shape)

        print('-- val_idxs:', meta_info['val_idxs'].shape)
        print('   - pos:', np.where(meta_info['all_labels'][meta_info['val_idxs']] == self.POS_LABEL)[0].shape)
        print('   - neg:', np.where(meta_info['all_labels'][meta_info['val_idxs']] == self.NEG_LABEL)[0].shape)

        test_idxs = meta_info['test_idxs']
        for subset in test_idxs:
            print('-- %s:' % subset, test_idxs[subset].shape)
            print('   - pos:', np.where(meta_info['all_labels'][test_idxs[subset]] == self.POS_LABEL)[0].shape)
            print('   - neg:', np.where(meta_info['all_labels'][test_idxs[subset]] == self.NEG_LABEL)[0].shape)

        print('')
        sys.stdout.flush()

    def feature_extraction(self, descriptor='RawImage', params='', n_jobs=1):

        start = get_time()

        # pre-load all images
        input_fnames = self.meta_info['all_fnames']
        tmpimgs = self.get_imgs(input_fnames)

        print('-- extracting ', descriptor, ' from raw images ...')
        tasks = []
        for idx in range(len(input_fnames)):
            tasks += [Extraction([], [],
                                 descriptor=descriptor,
                                 params=params,
                                 img=tmpimgs[idx])]

        resultimgs = []
        if n_jobs > 1:
            print("running %d tasks in parallel" % len(tasks))
            resultimgs += RunInParallel(tasks, n_jobs).run()
        else:
            print("running %d tasks in sequence" % len(tasks))
            for idx in range(len(input_fnames)):
                resultimgs += tasks[idx].run()
                progressbar('-- RunInSequence', idx, len(input_fnames))

        # replace the images with the extracted features
        self._imgs = np.array(resultimgs, dtype=np.uint8)

        elapsed = total_time_elapsed(start, get_time())
        print('spent time: {0}!'.format(elapsed))
        sys.stdout.flush()

    @staticmethod
    def list_dirs(root_path, file_types):
        folders = []

        for root, dirs, files in os.walk(root_path, followlinks=True):
            for f in files:
                if os.path.splitext(f)[1].lower() in [ft.lower() for ft in file_types]:
                    folders += [os.path.relpath(root, root_path)]
                    break

        return folders

    def meta_info_feats(self, output_path, file_types):
        return self._build_meta(output_path, file_types)

    @property
    def imgs(self):
        return self._imgs

    @imgs.setter
    def imgs(self, images):
        self._imgs = images

    @property
    def meta_info(self):
        try:
            return self.__meta_info
        except AttributeError:
            self.__meta_info = self._build_meta(self.dataset_path, self.file_types)
            return self.__meta_info

    @abstractmethod
    def _build_meta(self, in_path, file_types):
        pass
