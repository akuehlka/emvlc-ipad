# -*- coding: utf-8 -*-

import sys
import os
import tables
import time
import numpy as np
from antispoofing.mcnns.classification import *
from antispoofing.mcnns.features.descriptors import *


class Controller(object):
    def __init__(self, data, args):
        self.data = data
        self.args = args
        self.n_jobs = self.args.n_jobs

        self.features_path = "features"
        self.classification_path = "classification"
        self.path_to_features = os.path.join(self.data.output_path, self.args.descriptor, self.features_path)

    def _bsifExtraction(self, target, datasets, parameter_list):

        for i, ds in enumerate(datasets):
            clargs = parameter_list[i]

            # even raw image should go through, because the feature_extraction() method will load the images.

            # ATTENTION: class name is taken from the TARGET dataset
            descriptor_file = os.path.join('/afs/crc.nd.edu/group/cvrl/scratch_2/livdet/iris/tmp',
                                           '{}_{}{}.hdf5'.format(target.__class__.__name__.lower(),
                                                                 clargs['descriptor'],
                                                                 clargs['desc_params']))
            hdf5_lock = '{}.lock'.format(descriptor_file)

            if not os.path.exists(descriptor_file):
                print("-- extracting features ...")
                ds.feature_extraction(clargs['descriptor'],
                                      clargs['desc_params'],
                                      self.n_jobs)
                # save extracted features in a pytable
                # this way we'll be able to work with datasets that don't fit into the memory
                os.system('echo "1" > ' + hdf5_lock)
                hdf5_file = tables.open_file(descriptor_file, mode='w')
                filters = tables.Filters(complevel=5, complib='blosc')
                npimg = ds._imgs[0].astype(np.float32)
                data_storage = hdf5_file.create_earray(hdf5_file.root, 'bsiffeats',
                                                       tables.Atom.from_dtype(np.dtype(np.float32, npimg.shape)),
                                                       shape=tuple([0] + list(npimg.shape)),
                                                       filters=filters,
                                                       expectedrows=ds._imgs.shape[0])
                for img in ds._imgs:
                    data_storage.append(img[np.newaxis, :, :, :])
                hdf5_file.close()
                os.system('rm ' + hdf5_lock)

            while os.path.exists(hdf5_lock):
                print("Waiting for lock release...")
                time.sleep(5)

            print("-- loading pre-extracted features from {} ...".format(descriptor_file))
            data_storage = tables.open_file(descriptor_file, mode='r')
            ds._imgs = data_storage.root.bsiffeats

    def classification(self):

        start = get_time()

        algo = ml_algo[self.args.ml_algo]

        output_fname = "max_axis-{}-epochs-{}-bs-{}-losses-{}-lr-{}-decay-{}-optimizer-{}-reg-{}-fold-{}" \
            .format(self.args.max_axis,
                    self.args.epochs,
                    self.args.bs,
                    loss_functions[self.args.loss_function],
                    self.args.lr,
                    self.args.decay,
                    optimizer_methods[self.args.optimizer],
                    self.args.reg,
                    self.args.fold,
                    )
        
        cnn_ksize = (3, 3)
        if self.args.cnn_ksize > 0:
            output_fname += '-ksize-{}'.format(self.args.cnn_ksize)
            cnn_ksize = (self.args.cnn_ksize, self.args.cnn_ksize)

        output_path = os.path.join(self.data.output_path,
                                   self.args.descriptor,
                                   self.args.desc_params.replace(',', 'x'),
                                   self.classification_path,
                                   os.path.splitext(os.path.basename(self.args.permutation_path))[0],
                                   output_fname,
                                   )

        algo(output_path, self.data,
             input_shape=self.args.max_axis,
             epochs=self.args.epochs,
             batch_size=self.args.bs,
             loss_function=loss_functions[self.args.loss_function],
             lr=self.args.lr,
             decay=self.args.decay,
             optimizer=optimizer_methods[self.args.optimizer],
             regularization=self.args.reg,
             device_number=self.args.device_number,
             force_train=self.args.force_train,
             filter_vis=self.args.fv,
             layers_name=self.args.layers_name,
             fold=self.args.fold,
             ksize=cnn_ksize,
             ).run()

        elapsed = total_time_elapsed(start, get_time())
        s_time = []
        s_time.append('*******************************************')
        s_time.append('Partition : {0}, Permutation: {1}, Dataset: {2}.'.format(elapsed, 
                                                                        self.args.Permutation,
                                                                        str(self.data.__class__.__name__).lower()))
        s_time.append('Classification time: {0}!'.format(elapsed))
        s_time.append('*******************************************')
        for s in s_time:
            print(s)
        sys.stdout.flush()

    def execute_protocol(self):

        if self.args.feature_extraction:

            pl = {'descriptor': self.args.descriptor,
                  'desc_params': self.args.desc_params}
            self._bsifExtraction(self.data, [self.data], [pl])

        if self.args.classification:
            print("-- classifying ...")
            self.classification()
