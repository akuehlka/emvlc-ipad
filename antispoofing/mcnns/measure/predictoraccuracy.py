import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

from antispoofing.mcnns.utils.classifierselector import ClassifierSelector
import pdb


def readProtocolSummary():
    sum_file = 'output/protocol_summary.tmp'

    # update the summary
    CS = ClassifierSelector('output')
    summary = CS.summarize_results()
    with open(sum_file, 'w') as fout:
        w = csv.writer(fout)
        w.writerow(summary.dtype.names)
        w.writerows(summary)

    if os.path.exists(sum_file):
        df = pd.DataFrame(summary)
    else:
        raise "Summary file not found: {}".format(sum_file)

    # add a column with hter
    hter = pd.DataFrame({'hter': (df.apcer + df.bpcer) / 2})
    # add a column with # of features
    nfeatures = df.parameters.str.extract('\/man(?P<nfeatures>\d+)', expand=True)
    df = pd.concat([df, hter, nfeatures.astype(float, errors='ignore')], axis=1)

    return df


def selectBestPredictorOrder(dataset, df=None):

    if df == None:
        df = readProtocolSummary()

        # filter by dataset
        df = df[df.dsname == dataset]
        # filter weighted fusion classifier
        df = df[df.descriptor == 'weightedvotingfuser']
        # filter testgroup
        df = df[(df.testgroup == 'train_set') |
                (df.testgroup == 'val_set')]

    # select the best ordering for predictors according to the average
    dfimp = df[df.parameters.str.startswith('imp_as_weight/man')]
    meanimp = np.mean(
        (dfimp[dfimp.testgroup == 'train_set'].acc.values + dfimp[dfimp.testgroup == 'val_set'].acc.values) / 2)

    dfacc = df[df.parameters.str.startswith('acc_as_weight/man')]
    meanacc = np.mean(
        (dfacc[dfacc.testgroup == 'train_set'].acc.values + dfacc[dfacc.testgroup == 'val_set'].acc.values) / 2)

    if meanimp >= meanacc:
        df = dfimp
        weight = 'imp'
        print('Selecting IMPORTANCE to sort predictors')
    else:
        df = dfacc
        weight = 'acc'
        print('Selecting ACCURACY to sort predictors')

    return weight, df


def selectPredictorsByGain(dataset, maxpredictors=10, plot=False, weights='imp'):
    """
    Select predictors based on the cumulative gain of each predictor (discard negative gain)
    :param threshold:
    :return: pandas.DataFrame
    """

    df = readProtocolSummary()

    # filter by dataset
    df = df[df.dsname == dataset]
    # filter weighted fusion classifier
    df = df[df.descriptor == 'weightedvotingfuser']
    # filter testgroup
    df = df[(df.testgroup == 'train_set') |
            (df.testgroup == 'val_set')]

    # filter the weight type
    df = df[df.parameters.str.startswith('{}_as_weight/man'.format(weights))]

    diff = np.abs(df[df.testgroup == 'train_set'].acc.values - df[df.testgroup == 'val_set'].acc.values)
    mean = (df[df.testgroup == 'train_set'].acc.values + df[df.testgroup == 'val_set'].acc.values) / 2

    df = df.sort_values(['testgroup', 'nfeatures'])

    # calculate the gain for each predictor
    gains = mean[1:] - mean[:-1]

    # select only predictors that did not have loss
    predictors = np.where(gains >= 0)[0] + 1

    # predictor #0 is always there
    predictors = np.insert(predictors, 0, 0)

    # combine filters: predictors with gain AND lower difference between train/test
    goodpredictors = predictors[:maxpredictors]

    badpreds1 = np.where(gains < 0)[0] + 1
    badpreds2 = np.array([])

    if plot:
        plt.figure(figsize=(10,6))
        plt.plot(df[df.testgroup == 'train_set'].acc.values,'b-', alpha=0.5,label='Train')
        plt.plot(df[df.testgroup == 'val_set'].acc.values,'g-', alpha=0.5,label='Validation')
        plt.plot(mean,'r-',label='Mean')
        miny = plt.gca().get_ylim()[0]
        plt.plot(goodpredictors, np.ones_like(goodpredictors) * miny, 'ko', label='Selected predictors')
        plt.plot(badpreds1, np.ones_like(badpreds1) * miny, 'rx', label='Rejected: gain')
        plt.plot(badpreds2, np.ones_like(badpreds2) * miny, 'r+', label='Rejected: max preds')
        plt.grid()
        plt.legend()
        plt.title('Predictor selection - {}'.format(dataset.upper()))
        plt.xlabel('Predictors')
        plt.ylabel('Accuracy')
        plt.tight_layout()
        plt.show()

    return np.array(goodpredictors)
