# -*- coding: utf-8 -*-

import sys
import json
import tables
import time

from antispoofing.mcnns.datasets import *
from antispoofing.mcnns.classification import *
from antispoofing.mcnns.features.descriptors import *
from antispoofing.mcnns.controller import Controller
from antispoofing.mcnns.utils.misc import feature_selection, loadWeights, get_time
from antispoofing.mcnns.measure.predictoraccuracy import *
from antispoofing.mcnns.features.baseFeatureSelector import BaseFeatureSelector


class FusionController(Controller):
    
    def __init__(self, args):
        self.data = []

        # the first dataset object  is for the fusion classifier TARGET
        dataset = registered_datasets[args['dataset_nr']]
        data = dataset(args['dataset_path'],
                       ground_truth_path=args['ground_truth_path'],
                       permutation_path=args['permutation_path'],
                       iris_location=args['iris_location'],
                       output_path=args['output_path'],
                       operation=args['operation'],
                       max_axis=args['max_axis'],
                       augmentation=args['augmentation'], )
        data.output_path = os.path.join(args['output_path'],
                                        str(data.__class__.__name__).lower(),
                                        )
        data.hdf5_tmp_path = args['hdf5_tmp_path']
        self.data += [data]

        # next, we load one dataset object for each individual classifier (SOURCE predictors)
        for cl in args['pre-classifiers']['classifier_list']:
            # load global (default) parameters
            params = args['pre-classifiers']['global_params']
            # override default parameters with classifier-specific ones
            for k in cl:
                params[k] = cl[k]

            dataset = registered_datasets[params['dataset_nr']]
            data = dataset(params['dataset_path'],
                           ground_truth_path=params['ground_truth_path'],
                           permutation_path=params['permutation_path'],
                           iris_location=params['iris_location'],
                           output_path=params['output_path'],
                           operation=params['operation'],
                           max_axis=params['max_axis'],
                           augmentation=args['augmentation'],
                           )
            data.output_path = os.path.join(params['output_path'],
                                            str(data.__class__.__name__).lower(),
                                            )
            data.hdf5_tmp_path = args['hdf5_tmp_path']
            self.data += [data]

        self.args = args
        self.n_jobs = self.args['n_jobs']

        self.features_path = "features"
        self.classification_path = "classification"
        self.path_to_features = []
        for i, cl in enumerate(args['pre-classifiers']['classifier_list']):
            self.path_to_features += [os.path.join(self.data[i].output_path,
                                                   cl['descriptor'],
                                                   self.features_path)]

        self.crosseval = self.args['dataset_nr'] != self.args['pre-classifiers']['global_params']['dataset_nr']

    def _getPredictions(self, preds_file):

        # create a lock file
        lock_file = '{}.lock'.format(preds_file)
        os.system('echo "1" > {}'.format(lock_file))

        predictions = {}
        weight_factor = []
        parameters = self.args['pre-classifiers']['classifier_list']
        target_dataset = self.data[0]
        labels = target_dataset.meta_info['all_labels']

        # process feature extraction for source datasets
        self._bsifExtraction(target_dataset, self.data[1:], parameters)

        prediction_time = 0
        s_time = []

        # loop through all individual classifiers to be used
        for i, (cl, ds) in enumerate(zip(parameters, self.data[1:])):

            # load global (default) parameters
            params = self.args['pre-classifiers']['global_params']

            # override default parameters with classifier-specific ones
            for k in cl:
                params[k] = cl[k]
            pre_algo = ml_algo[params['algo_nr']]

            preclass_list = "max_axis-{}-epochs-{}-bs-{}-losses-{}-lr-{}-decay-{}-optimizer-{}-reg-{}-fold-{}".format(
                ds.max_axis,
                params['epochs'],
                params['bs'],
                loss_functions[params['loss_function']],
                params['lr'],
                params['decay'],
                optimizer_methods[params['optimizer']],
                params['reg'],
                params['fold']
            )

            # this is the path where we're gonna load the pre-classifier from
            pre_output_path = os.path.join(ds.output_path,
                                           params['descriptor'],
                                           params['desc_params'].replace(',', 'x'),
                                           self.classification_path,
                                           os.path.splitext(os.path.basename(params['permutation_path']))[0],
                                           preclass_list,
                                           )

            # read VALIDATION accuracy from this classifier
            weight_factor += [self.get_predictor_accs(pre_output_path)]

            # ***ATTENTION***: target dataset is loaded into the source classifiers
            data = ds._imgs

            # at this point, we have to load pytable data into an numpy array, otherwise we'll have trouble slicing
            # (within Keras)
            if type(data) is tables.earray.EArray:
                print(pre_output_path)
                tmp_data = np.zeros(data.shape, dtype=np.float32)
                for r in np.arange(data.shape[0]):
                    tmp_data[r, :, :, :] = data[r, :, :, :]
                data = tmp_data

            # get predictions for the entire dataset
            algo_tmp = pre_algo(pre_output_path, ds,
                                input_shape=ds.max_axis,
                                epochs=params['epochs'],
                                batch_size=params['bs'],
                                loss_function=loss_functions[
                                    params['loss_function']],
                                lr=params['lr'],
                                decay=params['decay'],
                                optimizer=optimizer_methods[params['optimizer']],
                                regularization=params['reg'],
                                device_number=self.args['device_number'],
                                force_train=self.args['force_train'],
                                fold=params['fold'],
                                )
            predictions[params['descriptor'] + str(i)] = algo_tmp.predict(data)

            s_time.append('Avg. prediction time: {}, {}, {}\n'.format(algo_tmp.prediction_time,
                                                                      params['desc_params'].replace(',', 'x'),
                                                                      str(self.data[0].__class__.__name__).lower()))
        dsname = str(ds.__class__.__name__).lower()
        with open( os.path.join(self.args['output_path'], 'time_{}.txt'.format(dsname) ), 'a' ) as f:
            f.writelines(s_time)

        # gather all pre-predictions in a matrix
        input_preds = []
        for i, cl in enumerate(self.args['pre-classifiers']['classifier_list']):
            input_preds += [predictions[cl['descriptor'] + str(i)]]
        input_preds = np.array(input_preds).T
        weight_factor = np.array(weight_factor)

        # save the pre-predictions, for next time
        np.savez(preds_file,
                 input_preds=input_preds,
                 clf_accuracy=weight_factor,
                 labels=labels)

        # release the lock
        os.system('rm ' + lock_file)

        return input_preds, weight_factor

    def classification(self):

        start = get_time()

        voting_algo = ml_algo[self.args['algo_nr']]
        use_feature_selection = 'featsel' in self.args and self.args['featsel'] != 'man'

        p_weighttype = ''
        weight_type_subfolder = ''
        if 'weighttype' in self.args:
            p_weighttype = self.args['weighttype']
            weight_type_subfolder = '{}_as_weight'.format(p_weighttype)

        # determine the best sorting order for predictors
        if p_weighttype == 'best':
            p_weighttype, _ = selectBestPredictorOrder(self.data[0].__class__.__name__.lower())
            weight_type_subfolder = '{}_as_weight'.format(p_weighttype)

        # define a file name to save intermediary predictions
        output_path = os.path.join(self.data[0].output_path,
                                   voting_algo.__name__.lower(),
                                   self.classification_path,
                                   weight_type_subfolder
                                   )
        # by default, use the SOURCE predictor's output directory
        preds_file = os.path.join('/'.join(output_path.split('/')[:-1]), 'predictions.npz')

        if self.crosseval:
            prefix_path = '/'.join(self.data[1].output_path.split('/')[:-1])
            crosseval_path = os.path.join(prefix_path,
                                   'cross_{}'.format(str(self.data[1].__class__.__name__).lower()), # livdetclarkson
                                   str(self.data[0].__class__.__name__).lower() # livdetww
                                   )
            output_path = crosseval_path
            preds_file = os.path.join(output_path, 'predictions.npz')

        if self.args['force_predict']:
            # save predictions in the TARGET output directory
            output_path = self.args['output_path']
            preds_file = os.path.join(output_path, 'predictions.npz')

        os.makedirs(output_path, exist_ok=True)

        # check for lock file/wait
        lock_file = '{}.lock'.format(preds_file)
        while os.path.exists(lock_file):
            print("Waiting for lock release...")
            time.sleep(5)

        if (os.path.exists(preds_file)) and \
           (not eval(self.args['force_train'])) and \
           (not self.args['force_predict']):
            # load the pre-calculated predictions
            npzfile = np.load(preds_file)
            input_preds = npzfile['input_preds']
            weight_factor = npzfile['clf_accuracy']
        else:
            # get predictions by all individual classifiers
            input_preds, weight_factor = self._getPredictions(preds_file)

        # these pre-predictions will be features of the dataset
        # initially, all features will be used
        self.data[0]._imgs = input_preds

        feats = np.arange(0, len(self.args["pre-classifiers"]["classifier_list"]))
        nfeats = 0
        if 'nfeats' in self.args:
            nfeats = self.args['nfeats']
        if 'features' in self.args:
            if nfeats == 0:
                nfeats = len(self.args['features'])
            feats = np.array(self.args['features'])

        featselpath = ""
        if 'featsel' in self.args:
            featselpath = self.args['featsel'] + str(nfeats)

        # redefine the output path accordingly
        output_path = os.path.join(self.data[0].output_path,
                                   voting_algo.__name__.lower(),
                                   self.classification_path,
                                   weight_type_subfolder,
                                   featselpath,
                                   os.path.basename(self.args['permutation_path'])
                                   )
        # cross-dataset output
        if self.crosseval:
            output_path = os.path.join(crosseval_path,
                                   voting_algo.__name__.lower(),
                                   weight_type_subfolder,        # imp_as_weight
                                   featselpath,                  # man1
                                   os.path.basename(self.args['permutation_path'])
                                   )
        os.makedirs(output_path, exist_ok=True)

        # now we can load the weights
        if weight_type_subfolder:

            # *** we should load weights from the source dataset 
            wf = loadWeights(self.data[1].__class__.__name__.lower(),
                                  weight_type_subfolder)

            nfeats_tmp = nfeats
            if nfeats_tmp == 0:
                nfeats_tmp = len(wf)

            print("Features loaded: ", wf[:, 0].astype(int))
            feats = wf[:nfeats_tmp, 0].astype(int)
            # we can't crop the weights vector before we have performed feature selection
            weight_factor = wf[:, 1]

        # feature selection
        if use_feature_selection:

            # perform feature selection on the training AND VALIDATION sets
            train_ixs = self.data[0].meta_info['train_idxs']
            val_ixs = self.data[0].meta_info['val_idxs']

            # *** here is where feature selection happens ***
            if self.args['featsel'] in ('fw', 'mi', 'rnd'):
                feats = feature_selection(self.data[0]._imgs[train_ixs, :, :, :],
                                          self.data[0].meta_info['all_labels'][train_ixs],
                                          self.args['featsel'])[:nfeats]
            else:
                fs = BaseFeatureSelector(algorithm=voting_algo,
                                         data=self.data[0]._imgs,
                                         labels=self.data[0].meta_info['all_labels'],
                                         trainix=train_ixs,
                                         valix=val_ixs)
                if self.args['featsel'] == 'rffw':
                    feats = fs.performFeatureSelection(features=feats, weights=weight_factor)
                elif self.args['featsel'] == 'corr':
                    feats = fs.performFeatureSelection(features=feats,
                                                       weights=weight_factor,
                                                       type='correlation')
            # *** end of feature selection ***

            # save the features that were selected
            json_fname = os.path.join(output_path, 'featsel_{}.json'.format(self.args['featsel']))
            with open(json_fname, mode='w') as f:
                print("--saving featsel.json file:", json_fname)
                sys.stdout.flush()
                f.write(json.dumps(feats.tolist(), indent=4))

        # after having selected predictors we keep only the respective weights
        weight_factor = weight_factor[feats]

        va = voting_algo(output_path, self.data[0])
        # restrict the predictors to be used.
        va.predictors = feats
        if type(va) is WeightedVotingFuser:
            va.weight_factor = weight_factor
        va.run()

        s_time = []
        s_time.append('Avg. prediction time: {}, {}\n'.format(va.prediction_time,
                                                              str(type(va))))
        dsname = str(self.data[0].__class__.__name__).lower()
        with open( os.path.join(self.args['output_path'], 'time_{}.txt'.format(dsname) ), 'a' ) as f:
            f.writelines(s_time)


        elapsed = total_time_elapsed(start, get_time())
        print('spent time: {0}!'.format(elapsed))
        sys.stdout.flush()

    def execute_protocol(self):

        if self.args['classification']:
            print("-- classifying ...")
            self.classification()

    def get_predictor_accs(self, path):
        # load the test accuracy for the specified path
        with open(path + '/val_set/results.json') as fp:
            accresults = json.load(fp)
        return accresults['0.5']['acc']
