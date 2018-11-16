# -*- coding: utf-8 -*-

import os
import sys
import json
import matplotlib.mlab as mlab
import numpy as np
import csv

from matplotlib import ticker
from matplotlib import pyplot as plt

from abc import ABCMeta, abstractmethod
from antispoofing.mcnns.utils import *
from antispoofing.mcnns.measure import *
from sklearn import metrics
from itertools import zip_longest


class BaseClassifier(metaclass=ABCMeta):
    """ This Class implements the common methods that will be use for all classifiers. """

    def __init__(self, output_path, dataset, fold=0, dataset_b=None):

        self.verbose = True
        self.output_path = os.path.abspath(output_path)
        self.dataset = dataset
        self.fold = fold
        self.dataset_b = dataset_b

    def interesting_samples(self, all_fnames, test_sets, class_report, predictions, threshold_type='EER'):
        """ This method persists a dictionary containing interesting samples for later visual assessments, which contains the filenames
        for samples that were incorrectly classified.

        Args:
            all_fnames (numpy.ndarray): A list containing the filename for each image in the dataset.
            test_sets (dict): A dictionary containing the data and the labels for all testing sets that compose the dataset.
            class_report (dict): A dictionary containing several evaluation measures for each testing set.
            predictions (dict): A dictionary containing the predicted scores and labels for each testing set.
            threshold_type (str): Defines what threshold will be considered for deciding the false acceptance and false rejections.

        """

        int_samples = {}
        predictions_test = predictions.copy()
        predictions_test.pop('train_set')
        predictions_test.pop('val_set')

        for key in predictions_test:

            gt = predictions_test[key]['gt']
            scores = predictions_test[key]['predicted_scores']
            test_idxs = test_sets[key]['idxs']
            int_samples_idxs = get_interesting_samples(gt, scores, class_report[key][threshold_type]['threshold'])

            int_samples[key] = {}

            for key_samples in int_samples_idxs.keys():
                int_samples[key][key_samples] = {'input_fnames': []}
                for idx in int_samples_idxs[key_samples]:
                    int_samples[key][key_samples]['input_fnames'] += [all_fnames[test_idxs[idx]]]

        json_fname = os.path.join(self.output_path, 'int_samples.json')
        with open(json_fname, mode='w') as f:
            print("--saving json file:", json_fname)
            sys.stdout.flush()
            f.write(json.dumps(int_samples, indent=4))

    def save_performance_results(self, class_report):
        """ Save the performance results in a .json file.

        Args:
            class_report (dict): A dictionary containing the evaluation results for each testing set.

        """

        print('-- saving the performance results in {0}\n'.format(self.output_path))
        sys.stdout.flush()

        for k in class_report:
            output_dir = os.path.join(self.output_path, k)
            try:
                os.makedirs(output_dir)
            except OSError:
                pass

            json_fname = os.path.join(output_dir, 'results.json')
            with open(json_fname, mode='w') as f:
                print("--saving results.json file:", json_fname)
                sys.stdout.flush()
                f.write(json.dumps(class_report[k], indent=4))

    def plot_score_distributions(self, thresholds, neg_scores, pos_scores, set_name):
        """ Plot the score distribution for a binary classification problem.

        Args:
            thresholds (list): A list of tuples containing the types and the values of the thresholds applied in this work.
            neg_scores (numpy.ndarray): The scores for the negative class.
            pos_scores (numpy.ndarray): The scores for the positive class.
            set_name (str): Name of the set used for computing the scores

        """

        plt.clf()
        plt.figure(figsize=(12, 10), dpi=100)

        plt.title("Score distributions (%s set)" % set_name)
        n, bins, patches = plt.hist(neg_scores, bins=25, normed=True, facecolor='green', alpha=0.5, histtype='bar',
                                    label='Negative class')
        na, binsa, patchesa = plt.hist(pos_scores, bins=25, normed=True, facecolor='red', alpha=0.5, histtype='bar',
                                       label='Positive class')

        # -- add a line showing the expected distribution
        y = mlab.normpdf(bins, np.mean(neg_scores), np.std(neg_scores))
        _ = plt.plot(bins, y, 'k--', linewidth=1.5)
        y = mlab.normpdf(binsa, np.mean(pos_scores), np.std(pos_scores))
        _ = plt.plot(binsa, y, 'k--', linewidth=1.5)

        for thr_type, thr_value in thresholds:
            plt.axvline(x=thr_value, linewidth=2, color='blue')
            plt.text(thr_value, 0, str(thr_type).upper(), rotation=90)

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.xlabel('Scores', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)

        plt.legend()

        filename = os.path.join(self.output_path, '%s.score.distribution.png' % set_name)
        plt.savefig(filename)

    @staticmethod
    def plot_crossover_error_rate(neg_scores, pos_scores, filename, n_points=1000):
        """ TODO: Not ready yet.

        Args:
            neg_scores (numpy.ndarray):
            pos_scores (numpy.ndarray):
            filename (str):
            n_points (int):
        """

        if len(neg_scores)==0 or len(pos_scores)==0:
            print("Not possible to calculate FAR/FRR curve when there's only one class.")
            return

        fars, frrs, thrs = farfrr_curve(neg_scores, pos_scores, n_points=n_points)
        x_range = np.arange(0, len(thrs), 1)

        # -- create the general figure
        fig1 = plt.figure(figsize=(12, 8), dpi=300)

        # -- plot the FAR curve
        ax1 = fig1.add_subplot(111)
        ax1.plot(fars[x_range], 'b-')
        plt.ylabel("(BPCER) FAR")

        # -- plot the FRR curve
        ax2 = fig1.add_subplot(111, sharex=ax1, frameon=False)
        ax2.plot(frrs[x_range], 'r-')
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        plt.ylabel("(APCER) FRR")

        # plt.xticks(x_range, thrs[x_range])
        plt.xticks()
        ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        plt.xlabel('Threshold')
        plt.ylabel('Error Rate')

        plt.show()
        plt.savefig(filename)

    @staticmethod
    def classification_results_summary(report):

        print('-- Classification Results')

        headers = ['Testing set', 'Threshold (value)', 'AUC', 'ACC', 'Balanced ACC', 'BPCER (FRR)', 'APCER (FAR)']
        header_line = "| {:<20s} | {:<20s} | {:<12s} | {:<12s} | {:<12s} | {:<12s} | {:<12s} |\n".format(*headers)
        sep_line = '-' * (len(header_line) - 1) + '\n'

        final_report = sep_line
        final_report += header_line

        for k1 in sorted(report):
            final_report += sep_line
            for k2 in sorted(report[k1]):
                values = [k1,
                          "{} ({:.2f})".format(k2, report[k1][k2]['threshold']),
                          report[k1][k2]['auc'],
                          report[k1][k2]['acc'],
                          report[k1][k2]['bacc'],
                          report[k1][k2]['bpcer'],
                          report[k1][k2]['apcer'],
                          ]
                line = "| {:<20s} | {:<20s} | {:<12.4f} | {:<12.4f} | {:<12.4f} | {:<12.4f} | {:<12.4f} |\n".format(
                    *values)
                final_report += line
        final_report += sep_line

        print(final_report)
        sys.stdout.flush()

    def performance_evaluation(self, predictions):
        """ Compute the performance of the fitted model for each test set.

        Args:
            predictions (dict): A dictionary with the ground-truth, the predicted scores and the predicted labels for the testing data.
            For example:
                {'test': {'gt': y_test,
                          'predicted_labels': y_pred,
                          'predicted_scores': y_scores,
                          }
                }

        Returns:
            dict: A dictionary containing the performance results for each testing set.
            For example:
                {'test': {'acc': acc,
                          'apcer': apcer,
                          'bpcer': bpcer,
                          }
                }

        """

        try:
            os.makedirs(self.output_path)
        except OSError:
            pass

        report = {}

        # -- compute the thresholds using the training set
        gt_train = predictions['train_set']['gt']
        pred_scores_train = predictions['train_set']['predicted_scores']
        genuine_scores_train, attack_scores_train = split_score_distributions(gt_train, pred_scores_train, label_neg=0,
                                                                              label_pos=1)

        far_thr = float(far_threshold(genuine_scores_train, attack_scores_train, far_at=0.01))
        eer_thr = float(eer_threshold(genuine_scores_train, attack_scores_train))

        thresholds = [('FAR@0.01', far_thr),
                      ('EER', eer_thr),
                      ('0.5', 0.5),
                      ]

        # -- plotting the score distribution for the training set
        self.plot_score_distributions(thresholds, genuine_scores_train, attack_scores_train, 'train')

        # -- compute the evaluation metrics for the test sets
        for key in predictions:
            report[key] = {}

            ground_truth = predictions[key]['gt']
            pred_scores = predictions[key]['predicted_scores']

            neg_scores, pos_scores = split_score_distributions(ground_truth, pred_scores, label_neg=0, label_pos=1)

            try:
                # -- compute the Area Under ROC curve
                roc_auc = metrics.roc_auc_score(ground_truth, pred_scores)
            except ValueError:
                # it'll give an error when there's only one class, hence the exception handling
                roc_auc = 0
                pass

            for thr_type, thr_value in thresholds:
                # -- compute the FAR and FRR
                # -- FAR (BPCER) is the rate of Genuine images classified as Presentation Attacks images
                # -- FRR (APCER) is the rate of Presentation Attack images classified as Genuine images
                # -- Note: Presentation Attacks images represent the Positive Class (label 1) are the
                # --       genuine images represent the Negative Class (0)
                bpcer, apcer = farfrr(neg_scores, pos_scores, thr_value)

                # -- compute the ACC and Balanced ACC
                acc = acc_threshold(ground_truth, pred_scores, thr_value, label_neg=0, label_pos=1)
                bacc = bacc_threshold(ground_truth, pred_scores, thr_value, label_neg=0, label_pos=1)

                # -- save the results in a dictionary
                report[key][thr_type] = {'auc': roc_auc, 'acc': acc, 'bacc': bacc, 'threshold': thr_value,
                                         'apcer': apcer, 'bpcer': bpcer,
                                         }

            # -- plotting the score distribution and the Crossover Error Rate (CER) graph
            self.plot_score_distributions(thresholds, neg_scores, pos_scores, key)
            self.plot_crossover_error_rate(neg_scores, pos_scores,
                                           filename=os.path.join(self.output_path, '%s.cer.png' % key))

        if self.verbose:
            self.classification_results_summary(report)

        return report

    def run_evaluation_protocol(self):
        """ This method implements the whole training and testing process considering the evaluation protocol defined for the dataset. """

        try:
            os.makedirs(self.output_path)
        except OSError:
            pass

        # -- get the sets of images according to the protocol evaluation defined in each dataset.
        dataset_protocol = self.dataset.protocol_eval(fold=self.fold)

        # -- starting the training process
        self.training(dataset_protocol['train_set']['data'], dataset_protocol['train_set']['labels'],
                      dataset_protocol['test_set']['test']['data'], dataset_protocol['test_set']['test']['labels'])

        # -- starting the testing process

        # -- compute the predicted scores and labels for the training set
        predictions = {
            'train_set': self.testing(dataset_protocol['train_set']['data'], dataset_protocol['train_set']['labels'])}

        # -- compute the predicted scores and labels for the validation set (if it exists)
        if 'val_set' in dataset_protocol.keys():
            predictions['val_set'] = self.testing(dataset_protocol['val_set']['data'], dataset_protocol['val_set']['labels'])

        # -- compute the predicted scores and labels for the testing sets
        for key in dataset_protocol['test_set']:
            predictions[key] = self.testing(dataset_protocol['test_set'][key]['data'],
                                            dataset_protocol['test_set'][key]['labels'])

        # -- estimating the performance of the classifier
        class_report = self.performance_evaluation(predictions)

        # -- saving the performance results
        self.save_performance_results(class_report)

        # -- saving the raw scores
        for key in predictions:
            if key == 'train_set':
                nameixs = self.dataset.meta_info['train_idxs']
            elif key == 'val_set':
                nameixs = self.dataset.meta_info['val_idxs']
            else:
                nameixs = self.dataset.meta_info['test_idxs'][key]
            output = np.array(list(zip(self.dataset.meta_info['all_fnames'][nameixs],
                                       predictions[key]['predicted_scores'],
                                       predictions[key]['predicted_labels'],
                                       predictions[key]['gt'])),
                              dtype=[('fname', 'U300'), ('pred_score', 'f8'), ('pred_labels', 'i8'), ('gt', 'i8')]
                              )
            with open(os.path.join(self.output_path, '%s.scores.txt' % key), 'w') as f:
                fw = csv.writer(f)
                fw.writerow(['fname', 'pred_score', 'pred_label', 'gt'])
                fw.writerows(output)
                # np.savetxt(os.path.join(self.output_path, '%s.scores.txt' % key),
                #            output,
                #            # header='fname;pred_score;pred_label',
                #            delimiter=',')

        # # -- create a mosaic for the positive and negative images.
        # all_pos_idxs = np.where(self.dataset.meta_info['all_labels'] == 1)[0]
        # all_neg_idxs = np.where(self.dataset.meta_info['all_labels'] == 0)[0]
        
        # # based on https://docs.python.org/3/library/itertools.html
        # def grouper(iterable, n, fillvalue=None):
        #     "Collect data into fixed-length chunks or blocks"
        #     # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
        #     args = [iter(iterable)] * n
        #     return list(zip_longest(*args, fillvalue=fillvalue))

        # for i, g in enumerate(grouper(self.dataset._imgs[all_pos_idxs, :, :, :], 50)):
        #     create_mosaic(g,
        #                 n_col=10,
        #                 output_fname=os.path.join(self.output_path, 'mosaic-pos-class-1_{}.jpeg'.format(i)))
        # for i, g in enumerate(grouper(self.dataset._imgs[all_neg_idxs, :, :, :], 50)):
        #     create_mosaic(g,
        #                 n_col=10,
        #                 output_fname=os.path.join(self.output_path, 'mosaic-neg-class-0_{}.jpeg'.format(i)))

        # -- saving the interesting samples for further analysis
        all_fnames = self.dataset.meta_info['all_fnames']
        self.interesting_samples(all_fnames, dataset_protocol['test_set'], class_report, predictions,
                                 threshold_type='EER')

    def run(self):

        try:
            os.makedirs(self.output_path)
        except OSError:
            pass

        self.run_evaluation_protocol()

    @abstractmethod
    def training(self, x_train, y_train, x_validation=None, y_validation=None):
        return NotImplemented

    @abstractmethod
    def testing(self, x_test, y_test):
        return NotImplemented

    @abstractmethod
    def predict(self, x_values):
        return NotImplemented
