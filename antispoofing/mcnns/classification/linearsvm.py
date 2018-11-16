# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pickle

from sklearn.svm import SVC, LinearSVC
from sklearn.grid_search import GridSearchCV
from antispoofing.mcnns.utils import *
from antispoofing.mcnns.classification.baseclassifier import BaseClassifier


class LinearSVM(BaseClassifier):
    """
    This class implements a classification scheme by using Support Vector Machine algorithm available on scikit-learn
    package. This implementation can deal with both multi-class and binary problems.
    """

    def __init__(self, output_path, dataset, fold=0,
                 input_shape=None, epochs=None, batch_size=None,
                 loss_function=None, lr=None, decay=None,
                 optimizer=None, regularization=None, device_number=None,
                 force_train=False, filter_vis=None, layers_name=None,
                 ksize=None):

        super(LinearSVM, self).__init__(output_path, dataset, fold=fold)

        self.dataset = dataset
        self.output_path = output_path
        self.output_model = os.path.join(self.output_path, "lin_svm_model.pkl")
        self.model = None

        self.num_classes = 2
        self.force_train = force_train

    def training(self, x_train, y_train, svm_params=None, debug=True):
        print('Training ...')
        sys.stdout.flush()

        if os.path.exists(self.output_model):
            # -- try loading model
            with open(self.output_model, 'rb') as pf:
                self.model = pickle.load(pf)
        else:
            self.model = None

        # -- True if model does not generated yet
        if not self.model:
            print('-- building model')
            sys.stdout.flush()

            # if not svm_params:
            #     # execute a grid search for good params

            categories = np.unique(y_train)

            # -- train a linear SVC
            self.model = LinearSVC()
            newshape = (x_train.shape[0], np.prod(x_train.shape[1:]))

            print('-- training ...')
            sys.stdout.flush()
            self.model.fit(x_train.reshape(newshape) , y_train)

            # -- save the model
            with open(self.output_model,'wb') as pf:
                pickle.dump(self.model, pf)

    def predict(self, x_values):
        if not self.model:
            # -- load the fitted classifier
            with open(self.output_model, 'rb') as pf:
                self.model = pickle.load(pf)

        newshape = (x_train.shape[0], np.prod(x_train.shape[1:]))

        new_preds = self.model.predict(x_values.reshape(newshape))

        return np.array(new_preds)


    def testing(self, x_test, y_test):

        print('Testing ...')
        sys.stdout.flush()

        outputs = {}
        if not self.model:
            # -- load the fitted classifier
            with open(self.output_model, 'rb') as pf:
                self.model = pickle.load(pf)

        if self.model:
            new_preds = []
            new_scores = []

            newshape = (x_test.shape[0], np.prod(x_test.shape[1:]))
            new_preds = self.model.predict(x_test.reshape(newshape))

            new_preds = np.array(new_preds)
            new_scores = np.array(new_preds)

            # -- define the output dictionary
            outputs = {'gt': y_test,
                    'predicted_labels': new_preds,
                    'predicted_scores': new_scores,
                    }


        else:
            sys.exit('-- model not found! Please, execute training again!')

        return outputs
