import pandas as pd
import numpy as np
import os
import re
import json
from glob import glob
from matplotlib import pyplot as plt
import pdb


class ClassifierSelector():
    def __init__(self, path='', datasets=[], descriptors=[], testgroups=[]):
        self.path = path
        self.datasets = datasets
        self.descriptors = descriptors
        self.testgroups = testgroups

    def summarize_results(self, verbose=True):
        data = []
        dsnames = self.datasets
        if len(dsnames) == 0:
            dsnames = ['']
        descriptors = self.descriptors
        if len(descriptors) == 0:
            descriptors = ['']
        testgroups = self.testgroups
        if len(testgroups) == 0:
            testgroups = ['train_set', 'val_set', 'test', 'unknown_test']

        # -------------------------------------------------------------------------------------------------------DATASET
        for dsname in dsnames:
            # ------------------------------------------------------------------------------------------------DESCRIPTOR
            for descriptor in descriptors:
                # --------------------------------------------------------------------------------------------TEST GROUP
                for tg in testgroups:
                    fname = '{path}/{dsname}/{descriptor}'.format(path=self.path,
                                                                  dsname=dsname,
                                                                  descriptor=descriptor)
                    cmd = 'find {subfolder} -name "results.json" | grep \'/{tg}/\''.format(subfolder=fname,
                                                                                           tg=tg)
                    # print(cmd)
                    outputfiles = os.popen(cmd).read().split('\n')
                    print("Output files found for ", tg, ":", len(outputfiles))
                    for of in outputfiles:
                        if not of:
                            continue
                        if verbose:
                            print(of)

                        # clear variables to make sure
                        ds = ''
                        descname = ''
                        subfolder = ''
                        tgroup = ''
                        classifiers = ''

                        expr = r"([^/]+)/*/([^/]+)/((\[.+\])/)*classification/(.*/)*(train_set|val_set|test|unknown_test)"
                        # extract info from the path
                        mo = re.search(
                            expr,
                            of,
                            re.U)
                        if mo:
                            # for g in mo.groups():
                            #     print(g)
                            ds = mo.group(1)

                            # temporary:
                            ds = ds.replace('_seed7', '')
                            if 'seed' in ds:
                                # ignore files from other permutation
                                continue

                            descname = mo.group(2)
                            if mo.group(4):
                                # add 0s to sort correctly as string
                                filtersize = [int(s) for s in mo.group(4)[
                                    1:-1].split('x')]
                                newfsize = ['{:02d}'.format(
                                    i) for i in filtersize]
                                descname += "[{}]".format('x'.join(newfsize))
                            subfolder = mo.group(5).strip(
                                '/') if mo.group(5) else ''
                            tgroup = mo.group(6)

                            # selected classifiers
                            classifiers = ''
                            if descname == 'votingfuser':
                                # read the file with the selected classifier ids
                                cls_file_mask = os.path.dirname(
                                    of) + '/../featsel_*.json'
                                cls_file = glob(cls_file_mask)
                                if len(cls_file) > 0:
                                    with open(cls_file[0], 'r') as jf:
                                        cls_list = json.load(jf)
                                        classifiers = '+'.join([str(s)
                                                                for s in cls_list])

                            # classifiers and weights
                            if descname == 'weightedvotingfuser':
                                # read the file with classifiers and weights
                                weights_file_mask = os.path.dirname(
                                    of) + '/../pred_weights.json'
                                w_file = glob(weights_file_mask)
                                if len(w_file) > 0:
                                    with open(w_file[0], 'r') as jf:
                                        w_list = json.load(jf)
                                        w_list['calc_weight'] = [
                                            round(x, 2) for x in w_list['calc_weight']]
                                        w_list['weight_factor'] = [
                                            round(x, 2) for x in w_list['weight_factor']]
                                        classifiers = json.dumps(w_list)

                            # read the data
                            with open(of, 'r') as jf:
                                d = json.load(jf)
                                apcer = d['0.5']['apcer']
                                bpcer = d['0.5']['bpcer']
                                acc = d['0.5']['acc']

                            data.append(
                                (ds, tgroup, descname, subfolder, apcer, bpcer, acc, classifiers))

        data = np.array(data,
                        dtype=[('dsname', 'U30'),
                               ('testgroup', 'U20'),
                               ('descriptor', 'U20'),
                               ('parameters', 'U200'),
                               ('apcer', 'f8'),
                               ('bpcer', 'f8'),
                               ('acc', 'f8'),
                               ('classifiers', 'U1500')])

        return data


    def get_classifiers(self, of):
        classifiers = ''
        # read the file with classifiers and weights
        weights_file_mask = os.path.dirname(of) + '/../pred_weights.json'
        w_file = glob(weights_file_mask)
        if len(w_file)>0:
            with open(w_file[0],'r') as jf:
                w_list = json.load(jf)
                w_list['calc_weight'] = [round(x,2) for x in w_list['calc_weight']]
                w_list['weight_factor'] = [round(x, 2) for x in w_list['weight_factor']]
                classifiers = json.dumps(w_list)
        return classifiers

    def summarize_results_new(self, verbose=True):
        data = []
        dsnames = self.datasets
        if len(dsnames) == 0:
            dsnames = ['']
        descriptors = self.descriptors
        if len(descriptors) == 0:
            descriptors = ['']
        testgroups = self.testgroups
        if len(testgroups) == 0:
            testgroups = ['train_set', 'val_set', 'test', 'unknown_test']

        # -------------------------------------------------------------------------------------------------------DATASET
        for dsname in dsnames:
            # ------------------------------------------------------------------------------------------------DESCRIPTOR
            for descriptor in descriptors:
                # --------------------------------------------------------------------------------------------TEST GROUP
                for tg in testgroups:
                    fname = '{path}/{dsname}/{descriptor}'.format(path=self.path,
                                                                  dsname=dsname,
                                                                  descriptor=descriptor)
                    cmd = 'find {subfolder} -name "results.json" | grep \'/{tg}/\''.format(subfolder=fname,
                                                                                           tg=tg)
                    # print(cmd)
                    outputfiles = os.popen(cmd).read().split('\n')
                    print("Output files found for ", tg, ":", len(outputfiles))
                    for of in outputfiles:
                        if not of:
                            continue
                        if verbose:
                            print(of)

                        # clear variables to make sure
                        ds = ''
                        descname = ''
                        subfolder = ''
                        tgroup = ''
                        classifiers = ''

                        expr = r"([^/]+)/*/([^/]+)/((\[.+\])/)*classification/(.*/)*(train_set|val_set|test|unknown_test)"
                        # extract info from the path
                        mo = re.search(
                            expr,
                            of,
                            re.U)
                        if mo:
                            # for g in mo.groups():
                            #     print(g)
                            ds = mo.group(1)

                            # temporary:
                            ds = ds.replace('_seed7','')
                            if 'seed' in ds:
                                # ignore files from other permutation
                                continue

                            descname = mo.group(2)
                            if mo.group(4):
                                # add 0s to sort correctly as string
                                filtersize = [int(s) for s in mo.group(4)[1:-1].split('x')]
                                newfsize = ['{:02d}'.format(i) for i in filtersize]
                                descname += "[{}]".format('x'.join(newfsize))
                            subfolder = mo.group(5).strip('/') if mo.group(5) else ''
                            tgroup = mo.group(6)

                            # selected classifiers
                            classifiers = ''
                            if descname=='votingfuser':
                                # read the file with the selected classifier ids
                                cls_file_mask = os.path.dirname(of) + '/../featsel_*.json'
                                cls_file = glob(cls_file_mask)
                                if len(cls_file)>0:
                                    with open(cls_file[0],'r') as jf:
                                        cls_list = json.load(jf)
                                        classifiers = '+'.join([str(s) for s in cls_list])

                            # classifiers and weights
                            if descname=='weightedvotingfuser':
                                classifiers = self.get_classifiers(of)

                        else:
                            # look for another pattern (cross-evaluation)
                            expr = r"(cross_\w+)/(\w+)(/\w+)?(/\w+)?/(train_set|val_set|test|unknown_test)"
                            # extract info from the path
                            mo = re.search(expr, of,re.U)
                            if mo:
                                ds = mo.group(1)+"_"+mo.group(2)
                                ds = ds.replace('livdet','')
                                descname = mo.group(3).strip('/') 
                                if mo.group(4):
                                    subfolder = mo.group(4).strip('/')
                                tgroup = mo.group(5)

                                # additional classifier info
                                if 'weightedvotingfuser' in descname:
                                    classifiers = self.get_classifiers(of)

                        # test again
                        if mo:
                            # read the data
                            with open(of, 'r') as jf:
                                d = json.load(jf)

                                livdet_apcer = d['0.5']['apcer']
                                livdet_bpcer = d['0.5']['bpcer']
                                livdet_acc = d['0.5']['acc']
                                livdet_thres = d['0.5']['threshold']

                                eer_apcer = d['EER']['apcer']
                                eer_bpcer = d['EER']['bpcer']
                                eer_acc = d['EER']['acc']
                                eer_thres = d['EER']['threshold']

                                far_apcer = d['FAR@0.01']['apcer']
                                far_bpcer = d['FAR@0.01']['bpcer']
                                far_acc = d['FAR@0.01']['acc']
                                far_thres = d['FAR@0.01']['threshold']

                            data.append((ds, tgroup, descname, subfolder,
                                         livdet_apcer, livdet_bpcer, livdet_acc, livdet_thres,
                                         eer_apcer, eer_bpcer, eer_acc, eer_thres,
                                         far_apcer, far_bpcer, far_acc, far_thres,
                                         classifiers))

        data = np.array(data,
                        dtype=[('dsname', 'U60'),
                               ('testgroup', 'U20'),
                               ('descriptor', 'U60'),
                               ('parameters', 'U200'),
                               ('apcer', 'f8'), ('bpcer', 'f8'), ('acc', 'f8'), ('thres','f8'),
                               ('eer_apcer', 'f8'), ('eer_bpcer', 'f8'), ('eer_acc', 'f8'), ('eer_thres','f8'),
                               ('far_apcer', 'f8'), ('far_bpcer', 'f8'), ('far_acc', 'f8'), ('far_thres','f8'),
                               ('classifiers', 'U1500')])

        return data

    def select_classifiers(self):
        # df = pd.read_csv('scripts/protocol_summary.csv')
        # summarize the results into a dataframe
        df = pd.DataFrame(self.summarize_results())

        # filter results for livdet datasets
        df1 = df[np.logical_and(np.logical_or(df['dsname'] == 'livdetiris17_nd', df['dsname'] == 'livdetiris17_warsaw'),
                                np.logical_and(df['descriptor'] != 'simplefuser', df['descriptor'] != 'votingfuser'))]
        avgs = df1.groupby(['dsname', 'testgroup', 'descriptor']).mean()

        datasets = avgs.index.levels[0]
        testgroups = avgs.index.levels[1]
        for ds in datasets:
            for tg in testgroups:
                x, y = avgs.loc[ds,tg].index.tolist(), avgs.loc[ds,tg].acc.tolist()
                fig, ax = plt.subplots()
                plt.title(ds)
                plt.bar(x, y)
                plt.ylim([min(y) - 0.02, 1])
                plt.xlabel(tg)
                plt.ylabel('Accuracy')
                labels = ax.get_xticklabels()
                plt.setp(labels, rotation=90)
                plt.tight_layout()
        plt.show()
