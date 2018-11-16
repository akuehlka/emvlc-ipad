# -*- coding: utf-8 -*-

import os
import itertools
import numpy as np
from glob import glob

from antispoofing.mcnns.datasets.dataset import Dataset
from antispoofing.mcnns.utils import *
from sklearn import model_selection


class NDCLD15(Dataset):

    def __init__(self, dataset_path, ground_truth_path='', permutation_path='', iris_location='',
                 output_path='./working', file_types=('.png', '.bmp', '.jpg', '.tiff'),
                 operation='crop', max_axis=320,
                 ):

        super(NDCLD15, self).__init__(dataset_path, output_path, iris_location, file_types, operation, max_axis)

        self.ground_truth_path = ground_truth_path
        self.verbose = True

    def _build_meta(self, inpath, filetypes):

        img_idx = 0

        all_fnames = []
        all_labels = []
        all_idxs = []
        train_idxs = []
        test_idxs = []

        subject_id = []
        sensor = []

        hash_img_id = {}

        SEQUENCE_ID_COL = 3
        SUBJECT_ID_COL = 8
        SENSOR_ID_COL = 36
        TAG_LIST_COL = 61
        CONTACTS_COL = 50

        gt_list, gt_hash = read_csv_file(self.ground_truth_path, sequenceid_col=SEQUENCE_ID_COL)
        all_tag_list = gt_list[:, TAG_LIST_COL]

        # -- The rules for getting the genuine samples is the union between:
        # -- (1) the samples that have the value 'No' in the CONTACTS column cells;
        # -- (2) the samples that have an empty value in the TAG_LIST column cells.
        no_contact_lenses_idxs = np.where(gt_list[:, CONTACTS_COL] == 'No')[0]
        empty_tag_list_cells_idxs = np.array([idx for idx, tag in enumerate(all_tag_list) if '' == tag])
        without_contact_lenses_idxs = np.unique(np.concatenate((empty_tag_list_cells_idxs, no_contact_lenses_idxs)))

        # -- The rules for getting the presentation attacks (texture contact lenses) samples is:
        # -- (1) the samples that have the value 'contacts-texture' or 'contacts-cosmetic' in the TAG_LIST column cells.
        contact_texture_idxs = np.array([idx for idx, tag in enumerate(all_tag_list) if 'contacts-texture' in tag or 'contacts-cosmetic' in tag])
        with_contact_lenses_idxs = contact_texture_idxs

        sequence_id_presentation_attack_class = gt_list[with_contact_lenses_idxs, SEQUENCE_ID_COL]
        sequence_id_genuine_class = gt_list[without_contact_lenses_idxs, SEQUENCE_ID_COL]

        # # -- for debugging ---------------------------------------------------------------------------------------------------------------
        # labeled_idxs = np.concatenate((with_contact_lenses_idxs, without_contact_lenses_idxs))
        # full_idxs = np.arange(len(gt_list))
        # nonlabeled_idxs = np.setdiff1d(full_idxs, labeled_idxs)
        #
        # self.LIV_DET_TRAIN = os.path.join(PROJECT_PATH, '../extra/LivDet-Iris-2017_splits/livdet-train.txt')
        # self.LIV_DET_TEST = os.path.join(PROJECT_PATH, '../extra/LivDet-Iris-2017_splits/livdet-test.txt')
        # self.LIV_DET_UNKNOWN_TEST = os.path.join(PROJECT_PATH, '../extra/LivDet-Iris-2017_splits/livdet-unknown_test.txt')
        #
        # liv_det_train_data, liv_det_train_hash = read_csv_file(self.LIV_DET_TRAIN, sequenceid_col=0, delimiter=' ')
        # liv_det_test_data, liv_det_test_hash = read_csv_file(self.LIV_DET_TEST, sequenceid_col=0, delimiter=' ')
        # liv_det_unknown_test_data, liv_det_unknown_test_hash = read_csv_file(self.LIV_DET_UNKNOWN_TEST, sequenceid_col=0, delimiter=' ')
        #
        # lvtrain_g = [os.path.splitext(l[0])[0] for l in liv_det_train_data if int(l[1])==0]
        # lvtrain_pa = [os.path.splitext(l[0])[0] for l in liv_det_train_data if int(l[1])==1]
        #
        # lvtest_g = [os.path.splitext(l[0])[0] for l in liv_det_test_data if int(l[1])==0]
        # lvtest_pa = [os.path.splitext(l[0])[0] for l in liv_det_test_data if int(l[1])==1]
        #
        # assert np.setdiff1d(np.array(lvtrain_g), sequence_id_genuine_class).size == 0
        # assert np.setdiff1d(np.array(lvtest_g), sequence_id_genuine_class).size == 0
        # assert np.setdiff1d(np.array(lvtrain_pa), sequence_id_presentation_attack_class).size == 0
        # assert np.setdiff1d(np.array(lvtest_pa), sequence_id_presentation_attack_class).size == 0
        # # --------------------------------------------------------------------------------------------------------------------------------

        folders = [self.list_dirs(inpath, filetypes)]
        folders = sorted(list(itertools.chain.from_iterable(folders)))

        non_labeled = []
        for i, folder in enumerate(folders):

            fnames = [glob(os.path.join(inpath, folder, '*' + filetype)) for filetype in filetypes]
            fnames = sorted(list(itertools.chain.from_iterable(fnames)))

            for j, fname in enumerate(fnames):

                # -- get the img_id from the image filename
                rel_path = os.path.relpath(fname, inpath)
                img_id, ext = os.path.splitext(os.path.basename(rel_path))
                img_id = img_id.split('_')[0]

                # -- check if the sample is labeled
                has_label = None
                if img_id in sequence_id_presentation_attack_class:
                    has_label = self.POS_LABEL
                elif img_id in sequence_id_genuine_class:
                    has_label = self.NEG_LABEL
                else:
                    non_labeled += [fname]

                # -- if the sample is labeled then we can use it
                if has_label is not None:

                    all_fnames += [fname]
                    all_idxs += [img_idx]

                    all_labels += [has_label]

                    subject_id += [gt_list[gt_hash[img_id]][SUBJECT_ID_COL]]
                    sensor += [gt_list[gt_hash[img_id]][SENSOR_ID_COL]]

                    hash_img_id[img_id] = img_idx

                    img_idx += 1

        all_fnames = np.array(all_fnames)
        all_labels = np.array(all_labels)
        all_idxs = np.array(all_idxs)
        subject_id = np.array(subject_id)
        sensor = np.array(sensor)

        non_labeled = np.array(non_labeled)
        print('-- non_labeled:', non_labeled.shape)
        np.savetxt('non_labeled.txt', non_labeled, fmt='%s')

        # unique_subject_id = np.unique(subject_id)
        # unique_subject_id_labels = np.concatenate([all_labels[np.where(s_id == subject_id)[0]][:1] for s_id in unique_subject_id])
        # unique_subject_id_len = np.array([len(np.unique(all_labels[np.where(s_id == subject_id)[0]])) for s_id in unique_subject_id])
        # # unique_subject_id_duplicate = np.array([s_id for s_id in unique_subject_id if len(np.unique(all_labels[np.where(s_id == subject_id)[0]]))==2])
        # # subject_id_duplicate_idxs = np.array([np.where(dup == subject_id)[0] for dup in unique_subject_id_duplicate])
        #
        # labels = []
        # for s_id in unique_subject_id:
        #     label = all_labels[np.where(s_id == subject_id)[0]][0]
        #     labels += [label]
        #     print(label)
        # pdb.set_trace()

        r_dict = {'all_fnames': all_fnames,
                  'all_labels': all_labels,
                  'all_idxs': all_idxs,
                  'subject_id': subject_id,
                  'sensor': sensor,
                  'hash_img_id': hash_img_id,
                  }

        return r_dict

    def protocol_eval(self, fold=0, n_fold=5, train_size=0.5):

        # -- loading the training data and its labels
        all_fnames = self.meta_info['all_fnames']
        all_labels = self.meta_info['all_labels']
        subject_id = self.meta_info['subject_id']
        all_data = self.get_imgs(all_fnames)

        unique_subject_id = np.unique(subject_id)
        unique_subject_id_labels = np.concatenate([all_labels[np.where(s_id == subject_id)[0]][:1] for s_id in unique_subject_id])

        assert unique_subject_id.size == unique_subject_id_labels.size, 'It is something wrong with unique_subject_id array'

        rand_state = np.random.RandomState(7)
        sss = model_selection.StratifiedShuffleSplit(n_splits=n_fold, test_size=0.5, random_state=rand_state)

        folds_subject_id_idxs = []
        for train_index, test_index in sss.split(unique_subject_id, unique_subject_id_labels):
            folds_subject_id_idxs += [[train_index, test_index]]
            folds_subject_id_idxs += [[test_index, train_index]]

        subject_id_train = unique_subject_id[folds_subject_id_idxs[fold][0]]
        subject_id_test = unique_subject_id[folds_subject_id_idxs[fold][1]]

        if self.verbose:
            print('-- Total of samples:', unique_subject_id.shape)
            print('-- subject_id_train size:', subject_id_train.shape)
            print('-- subject_id_test size:', subject_id_test.shape)

        n_intersected_samples = np.intersect1d(subject_id_train, subject_id_test).size

        try:
            assert n_intersected_samples == 0
        except AssertionError:
            raise Exception('Subject overlapping between training and testing sets: %d', n_intersected_samples)

        train_idxs = np.concatenate([np.where(s_id == subject_id)[0] for s_id in subject_id_train])
        test_idxs = {'test': np.concatenate([np.where(s_id == subject_id)[0] for s_id in subject_id_test])}

        train_set = {'data': all_data[train_idxs], 'labels': all_labels[train_idxs], 'idxs': train_idxs}

        test_set = {}
        for test_id in test_idxs:
            if test_idxs[test_id].size:
                test_set[test_id] = {'data': all_data[test_idxs[test_id]],
                                     'labels': all_labels[test_idxs[test_id]],
                                     'idxs': test_idxs[test_id],
                                     }

        if self.verbose:
            r_dict = {'train_idxs': train_idxs,
                      'test_idxs': test_idxs,
                      }

            meta_info = self.meta_info.copy()
            meta_info.update(r_dict)

            self.info(meta_info)

        return {'train_set': train_set, 'test_set': test_set}
