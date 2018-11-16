import os
import csv
import numpy as np
from antispoofing.mcnns.classification.baseclassifier import BaseClassifier

class SimpleFuser(BaseClassifier):

    def __init__(self, output_path, dataset, fold=0):

        super(SimpleFuser, self).__init__(output_path, dataset,
                                  fold=fold,
                                  )

        self.verbose = True

        self.dataset = dataset
        self.output_path = output_path
        self.output_model = os.path.join(self.output_path, "full_model.hdf5")
        self.output_weights = os.path.join(self.output_path, "weights.hdf5")

        self.num_classes = 2

        self.predictor_order =[]

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
                                      dataset_protocol['train_set']['labels'],
                                      [0, 1, 2])
        }

        # -- compute the predicted scores and labels for the testing sets
        for key in dataset_protocol['test_set']:
            predictions[key] = self.testing(dataset_protocol['test_set'][key]['data'],
                                            dataset_protocol['test_set'][key]['labels'],
                                            [0, 1, 2])

        # -- estimating the performance of the classifier
        class_report = self.performance_evaluation(predictions)

        # -- saving the performance results
        self.save_performance_results(class_report)

        # -- saving the raw scores
        for key in predictions:
            output = np.array(list(zip(self.dataset.meta_info['all_fnames'][self.dataset.meta_info['test_idxs']],
                                       predictions[key]['predicted_scores'],
                                       predictions[key]['predicted_labels'],
                                       predictions[key]['gt'])),
                              dtype=[('fname','U100'),('pred_score','f8'),('pred_labels','i8'),('gt','i8')]
                               )
            with open(os.path.join(self.output_path, '%s.scores.txt' % key), 'w') as f:
                fw = csv.writer(f)
                fw.writerow(output.dtype.names)
                fw.writerows(output)

        # # -- saving the interesting samples for further analysis
        # all_fnames = self.dataset.meta_info['all_fnames']
        # self.interesting_samples(all_fnames, dataset_protocol['test_set'], class_report, predictions, threshold_type='EER')


    def testing(self, pred_test, gt_test, pref_pred_order):
        """ This method is responsible for testing the performance of the result fusion.

        Args:
            pred_test (numpy.ndarray): Testing predictions by the individual classifiers. It should be a matrix of MxN,
                                       where M is the number of samples and N is the number of classifiers.
            gt_test (numpy.ndarray):   Labels of the Testing data
            preferred_predictor (int): Column indexes of the preferred predictors in order of importance, to be used
                                       when there's disagreement between the predictors

        Returns:
            A dictionary with the ground-truth, the predicted scores and the predicted labels for the testing data.
            For example:
            {'gt': y_test,
             'predicted_labels': y_pred,
             'predicted_scores': y_scores,
            }

        """
        predictors = pref_pred_order

        # start with the first predictor
        new_preds = pred_test[:, predictors[0]]
        new_scores = new_preds.astype(float)

        # a sigmoid function to penalize the confidence in scores that are changed
        sigm = lambda x: 1/(1+np.exp(-((x-0.5)*12)))

        # apply corrections to the disagreements
        for j in predictors[1:]:
            # find disagreements
            disagree = new_preds != pred_test[:, j]
            if len(disagree) > 0:
                new_preds[disagree] = pred_test[disagree, j]
                new_scores[disagree] = sigm(new_scores[disagree])


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