import os
import csv
import numpy as np
import json
from antispoofing.mcnns.classification.baseclassifier import BaseClassifier
import pdb


class WeightedVotingFuser(BaseClassifier):
    def __init__(self, output_path, dataset, predictors=[], weight_factor=[], fold=0):

        super(WeightedVotingFuser, self).__init__(output_path, dataset,
                                                  fold=fold,
                                                  )

        self.verbose = True

        self.dataset = dataset
        self.output_path = output_path
        self.output_model = os.path.join(self.output_path, "full_model.hdf5")
        self.output_weights = os.path.join(self.output_path, "weights.hdf5")

        self.num_classes = 2

        self.predictors = predictors
        self.weight_factor = weight_factor

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
            predictions['val_set'] = self.testing(dataset_protocol['val_set']['data'],
                                                  dataset_protocol['val_set']['labels'])

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

    def testing(self, predictor_matrix, gt):
        """ This method is responsible for testing the performance of the result fusion by voting among the pre-classifiers

        Args:
            predictor_matrix (numpy.ndarray): Testing predictions by the individual classifiers. It should be a matrix of MxN,
                                              where M is the number of samples and N is the number of classifiers.
            gt (numpy.ndarray):               Labels of the Testing data
            weight_factor (numpy.ndarray):         Accuracy of predictors (used to calculate the weights)


        Returns:
            A dictionary with the ground-truth, the predicted scores and the predicted labels for the testing data.
            For example:
            {'gt': y_test,
             'predicted_labels': y_pred,
             'predicted_scores': y_scores,
            }

        """
        # use all predictors available, unless they are specified
        sel_predictors = list(range(predictor_matrix.shape[1]))
        if len(self.predictors) > 0:
            sel_predictors = self.predictors

        # make sure all selected predictors have an assigned accuracy
        nselpreds = len(sel_predictors)
        nweightfacs = len(self.weight_factor)
        if nselpreds > nweightfacs:
            raise Exception('The number of selected predictors and the size of the list of accuracies do not match.')
        elif nselpreds < nweightfacs:
            self.weight_factor = self.weight_factor[:nselpreds]

        # calculate BWWV weights (as described in Moreno-Seco et al. 2006)
        order = np.argsort(self.weight_factor)
        w = np.linspace(0.01, 1, len(order))
        w = w[order]

        # save a list of predictors and their weights
        with open(os.path.join(self.output_path, 'pred_weights.json'), 'w') as f:
            pred_weights = {'predictors': sel_predictors.tolist(),
                            'weight_factor': self.weight_factor.tolist(),
                            'calc_weight': w.tolist()}
                            
            json.dump(pred_weights, f, indent=2)

        new_preds = []
        new_scores = []

        for sample in predictor_matrix:
            votes_pos = w[sample[sel_predictors]>0].sum()
            votes_neg = w[sample[sel_predictors]==0].sum()

            # default vote is negative (in case of tie)
            new_preds.append(1 if votes_pos > votes_neg else 0)
            # we're not producing scores, so just produce a label here
            new_scores.append(1. if votes_pos > votes_neg else 0.)

        new_preds = np.array(new_preds)
        new_scores = np.array(new_scores)

        # -- define the output dictionary
        r_dict = {'gt': gt,
                  'predicted_labels': new_preds,
                  'predicted_scores': new_scores,
                  }

        return r_dict

    def predict(self, x_values):
        return NotImplemented

    def training(self, x_train, y_train, x_validation=None, y_validation=None):
        return NotImplemented
