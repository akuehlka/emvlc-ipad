import os
import csv
import numpy as np
from antispoofing.mcnns.utils.misc import get_time
from antispoofing.mcnns.classification.baseclassifier import BaseClassifier
import pdb


class VotingFuser(BaseClassifier):
    def __init__(self, output_path, dataset, predictors=[], fold=0):

        super(VotingFuser, self).__init__(output_path, dataset,
                                          fold=fold,
                                          )

        self.verbose = True

        self.dataset = dataset
        self.output_path = output_path
        self.output_model = os.path.join(self.output_path, "full_model.hdf5")
        self.output_weights = os.path.join(self.output_path, "weights.hdf5")

        self.num_classes = 2

        self.predictors = predictors

        self.prediction_time = 0

    def run_evaluation_protocol(self):
        """ This method implements the evaluation of results based on the fusion of individual classifiers. """

        try:
            os.makedirs(self.output_path)
        except OSError:
            pass

        # -- get the sets of images according to the protocol evaluation defined in each dataset.
        dataset_protocol = self.dataset.protocol_eval(fold=self.fold)

        # -- starting the testing process

        # -- compute the predicted scores and labels for the training set
        predictions = {
            'train_set': self.testing(dataset_protocol['train_set']['data'],
                                      dataset_protocol['train_set']['labels'])
        }

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
                              dtype=[('fname', 'U100'), ('pred_score', 'f8'), ('pred_labels', 'i8'), ('gt', 'i8')]
                              )
            with open(os.path.join(self.output_path, '%s.scores.txt' % key), 'w') as f:
                fw = csv.writer(f)
                fw.writerow(output.dtype.names)
                fw.writerows(output)

                # # -- saving the interesting samples for further analysis
                # all_fnames = self.dataset.meta_info['all_fnames']
                # self.interesting_samples(all_fnames, dataset_protocol['test_set'], class_report, predictions, threshold_type='EER')

    def testing(self, pred_test, gt_test):
        """ This method is responsible for testing the performance of the result fusion by voting among the pre-classifiers

        Args:
            pred_test (numpy.ndarray): Testing predictions by the individual classifiers. It should be a matrix of MxN,
                                       where M is the number of samples and N is the number of classifiers.
            gt_test (numpy.ndarray):   Labels of the Testing data

        Returns:
            A dictionary with the ground-truth, the predicted scores and the predicted labels for the testing data.
            For example:
            {'gt': y_test,
             'predicted_labels': y_pred,
             'predicted_scores': y_scores,
            }

        """
        # use all predictors available, unless they are specified
        predictors = list(range(pred_test.shape[1]))
        if len(self.predictors) > 0:
            predictors = self.predictors

        new_preds = []
        new_scores = []

        start = get_time()

        for sample in pred_test:
            votes = len(sample[predictors])
            votes_pos = np.count_nonzero(sample[predictors])
            votes_neg = np.count_nonzero(np.logical_not(sample[predictors]))

            # default vote is negative (in case of tie)
            new_preds.append(1 if votes_pos > votes_neg else 0)
            # we're not using scores, so all we produce is a label
            new_scores.append(1. if votes_pos > votes_neg else 0.)

        new_preds = np.array(new_preds)
        new_scores = np.array(new_scores)

        elapsed = get_time() - start
        self.prediction_time = elapsed.seconds / pred_test.shape[0]

        # -- define the output dictionary
        r_dict = {'gt': gt_test,
                  'predicted_labels': new_preds,
                  'predicted_scores': new_scores,
                  }

        return r_dict

    def predict(self, x_values):
        return NotImplemented

    def training(self, x_train, y_train, x_validation=None, y_validation=None):
        return NotImplemented
