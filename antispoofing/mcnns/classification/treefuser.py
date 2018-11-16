import os
import csv
import sys
import numpy as np
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from antispoofing.mcnns.classification.baseclassifier import BaseClassifier
import pdb


class TreeFuser(BaseClassifier):
    def __init__(self, output_path, dataset, predictors=[], fold=0, force_train=False):

        super(TreeFuser, self).__init__(output_path, dataset,
                                          fold=fold,
                                          )

        self.verbose = True

        self.dataset = dataset
        self.output_path = output_path
        self.output_model = os.path.join(self.output_path, "full_model.pkl")
        self.output_weights = os.path.join(self.output_path, "weights.pkl")
        self.clf = None
        self.force_train = force_train

        self.num_classes = 2

        self.predictors = predictors

    def run_evaluation_protocol(self):
        """ This method implements the evaluation of results based on the fusion of individual classifiers. """

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

        # -- load the fitted classifier
        with open(self.output_model, 'rb') as pf:
            self.clf = pickle.load(pf)

        new_preds = []
        new_scores = []

        new_preds = self.clf.predict(pred_test)

        new_preds = np.array(new_preds)
        new_scores = np.array(new_preds)

        # -- define the output dictionary
        r_dict = {'gt': gt_test,
                  'predicted_labels': new_preds,
                  'predicted_scores': new_scores,
                  }

        return r_dict

    def predict(self, x_values):
        # -- load the fitted classifier
        with open(self.output_model, 'rb') as pf:
            self.clf = pickle.load(pf)

        new_preds = self.clf.predict(x_values)

        return np.array(new_preds)


    def training(self, x_train, y_train, x_validation=None, y_validation=None):
        """ This method implements the training process of our Random trees classifier.

        Args:
            x_train (numpy.ndarray): Training data
            y_train (numpy.ndarray): Labels of the training data
            x_validation (:obj: `numpy.ndarray`, optional): Testing data. Defaults to None.
            y_validation (:obj: `numpy.ndarray`, optional): Labels of the testing data. Defaults to None.

        """

        if self.force_train or not os.path.exists(self.output_model):
            print('-- training Random Forest Classifier...')
            sys.stdout.flush()

            # run a grid search to find best parameters
            rf = RandomForestClassifier()
            search_params = {"n_estimators": np.arange(1, 1000,100).tolist(),
                             "max_features": np.arange(1, 50, 2).tolist(),
                             "min_samples_leaf": np.arange(1, 50, 2).tolist()}
            grid = GridSearchCV(rf, param_grid=search_params, n_jobs=-1)
            grid.fit(x_train, y_train)
            best_params = grid.best_params_

            # -- fit the model with the best parameters
            self.clf = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                              max_features=best_params['max_features'],
                                              min_samples_leaf=best_params['min_samples_leaf'],
                                              n_jobs=-1)
            self.clf.fit(x_train, y_train)

            # -- save the tree importances
            pimpfile = os.path.join(self.output_path, 'predictor_importances.json')
            print("-- saving predictor importances:", pimpfile)
            sys.stdout.flush()
            with open(pimpfile, 'w') as f:
                json.dump({'predictor_importances': self.clf.feature_importances_.tolist()}, f, indent=2)

            # -- save the fitted model
            print("-- saving model", self.output_model)
            sys.stdout.flush()

            with open(self.output_model,'wb') as pf:
                pickle.dump(self.clf, pf)
        else:
            print('-- model already exists in', self.output_model)
            sys.stdout.flush()
