import numpy as np
import os
import pandas as pd
import pdb

from antispoofing.mcnns.classification import *


class BaseFeatureSelector(object):
    def __init__(self, algorithm, data, labels, trainix, valix):

        # create a dummy output path for the classifier
        output_dir = "/tmp/fs/"
        os.makedirs(output_dir, exist_ok=True)

        # instantiate the classifier
        self.va = algorithm(output_dir, data)

        # store important stuff
        self.data = data
        self.traindata = data[trainix]
        self.trainlabels = labels[trainix]

        self.valdata = data[valix]
        self.vallabels = labels[valix]

    def sortFeaturesByCorrelation(self, weights):

        # calculate correlation matrix using kendall method
        # (apparently it is more appropriate for non-continuous variables)
        predictors = pd.DataFrame(self.data)
        correlation = predictors.corr('kendall')

        # get the accuracy order of predictors
        pred_acc_order = np.argsort(weights)[::-1]
        most_important_predictor = pred_acc_order[0]

        # get correlation coefficients with regard to the first predictor (the more accurate/important)
        correlated = correlation[most_important_predictor]

        sel_preds = []
        sel_preds.append(pred_acc_order[most_important_predictor])
        corr_order = np.argsort(correlated[correlated != 1])
        for p in pred_acc_order[corr_order]:
            sel_preds.append(p)

        return np.array(sel_preds), weights[sel_preds]

    def performFeatureSelection(self, features, weights, type='default'):
        """Perform forward feature selection on the ordered feature set."""

        print("Selecting features...")

        if type == 'correlation':
            # sort features by correlation with the 1st (more important) feature
            features, weights = self.sortFeaturesByCorrelation(weights)

        # create a map for weights
        wmap = {i: w for i, w in zip(features, weights)}

        baseline_acc = 0
        features = features.tolist()
        cand_feats = []
        remain_feats = features

        # feature 0 is always selected
        selected_f = 0

        # repeat while we still have features
        while len(remain_feats) > 0:

            # initialize the set of features
            if selected_f > -1:
                cand_feats.append(remain_feats.pop(selected_f))
            selected_f = -1

            for i, f in enumerate(remain_feats):

                # prepare the classifier
                self.va.predictors = np.array(cand_feats + [f])
                if self.va.__class__.__name__ == 'WeightedVotingFuser':
                    self.va.weight_factor = np.array([wmap[w] for w in self.va.predictors])

                pred_train = self.va.testing(self.traindata, self.trainlabels)
                acc_train = np.sum(pred_train['predicted_labels'] == pred_train['gt']) / len(self.trainlabels)
                pred_val = self.va.testing(self.valdata, self.vallabels)
                acc_val = np.sum(pred_val['predicted_labels'] == pred_val['gt']) / len(self.vallabels)
                # acc = (acc_train + acc_val) / 2
                acc = acc_val

                # if mean accuracy improved, keep this feature
                if acc > baseline_acc:
                    baseline_acc = acc
                    selected_f = i

            # if none of the features was able to increase accuracy, stop
            if selected_f == -1:
                break
            print(cand_feats, "Accuracy:", baseline_acc)

        return np.array(cand_feats)
