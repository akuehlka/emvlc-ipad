# -*- coding: utf-8 -*-

import os
import sys
import json
import csv
from itertools import zip_longest

from antispoofing.mcnns.utils.common_imports import *
from antispoofing.mcnns.utils.misc import create_mosaic
from antispoofing.mcnns.classification.baseclassifier import BaseClassifier
from sklearn.utils import class_weight
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Flatten, Dense, Input

# from vis.utils import utils
# from vis.visualization import visualize_activation
# from vis.visualization import visualize_saliency


class VGG(BaseClassifier):
    def __init__(self, output_path, dataset, input_shape=200, epochs=50,
                 batch_size=8, loss_function=0, lr=0.01, decay=0.0005, optimizer='SGD', regularization=0.1,
                 device_number=0, force_train=False, filter_vis=False, layers_name=('conv_1',),
                 ksize=(3,3),
                 fold=0):

        super(VGG, self).__init__(output_path, dataset,
                                  fold=fold,
                                  )

        self.verbose = True

        self.dataset = dataset
        self.output_path = output_path
        self.output_model = os.path.join(self.output_path, "full_model.hdf5")
        self.output_weights = os.path.join(self.output_path, "weights.hdf5")
        self.model = None

        self.input_shape = (input_shape, input_shape, 1)
        self.num_classes = 2
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.lr = lr
        self.decay = decay
        self.optimizer = optimizer
        self.regularization = regularization
        self.device_number = device_number
        self.force_train = force_train
        self.filter_vis = filter_vis
        self.layers_name = list(layers_name)
        self.ksize = ksize

    def set_gpu_configuration(self):
        """
        This function is responsible for setting up which GPU will be used during the processing and some configurations
        related to GPU memory usage when the TensorFlow is used as backend.
        """

        if 'tensorflow' in keras.backend.backend():
            os.environ["CUDA_VISIBLE_DEVICES"] = self.device_number

            K.clear_session()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95, allow_growth=True, allocator_type='BFC')
            K.set_session(K.tf.Session(config=K.tf.ConfigProto(gpu_options=gpu_options,
                                                               allow_soft_placement=True,
                                                               log_device_placement=True)))

    def architecture_definition(self):
        """ In this method we define the CNN architecture """

        input_shape = (224,224,3)
        vgg = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        print('VGG16 model loaded.')

        # freeze the convolutional layers
        for layer in vgg.layers:
            layer.trainable = False

        x = Flatten(name='flatten')(vgg.layers[-1].output)
        x = Dense(4096, activation='relu', kernel_initializer='glorot_normal', name='fc1')(x)
        x = Dense(4096, activation='relu', kernel_initializer='glorot_normal', name='fc2')(x)
        x = Dense(2, activation='softmax', name='predictions')(x)

        self.model = Model(vgg.input, x, name='VGG16')

        if self.verbose:
            print(self.model.summary())

        # -- saving the CNN architecture definition in a .json file
        model_json = json.loads(self.model.to_json())
        json_fname = os.path.join(self.output_path, 'model.json')
        with open(json_fname, mode='w') as f:
            print("--saving json file:", json_fname)
            sys.stdout.flush()
            f.write(json.dumps(model_json, indent=4))

    def saving_training_history(self, history):

        # -- save the results obtained during the training process
        json_fname = os.path.join(self.output_path, 'training.history.json')
        with open(json_fname, mode='w') as f:
            print("--saving json file:", json_fname)
            sys.stdout.flush()
            f.write(json.dumps(history.history, indent=4))

        output_history = os.path.join(self.output_path, 'training.history.png')
        fig1 = plt.figure(figsize=(8, 6), dpi=100)
        title_font = {'size': '18', 'color': 'black', 'weight': 'normal', 'verticalalignment': 'bottom'}
        axis_font = {'size': '14'}
        font_size_axis = 12
        title = "Training History"

        plt.clf()
        plt.plot(range(1, len(history.history['acc']) + 1), history.history['acc'], color=(0, 0, 0), marker='o', linestyle='-', linewidth=2,
                 label='train')
        plt.plot(range(1, len(history.history['val_acc']) + 1), history.history['val_acc'], color=(0, 1, 0), marker='s', linestyle='-',
                 linewidth=2, label='test')

        plt.xlabel('Epochs', **axis_font)
        plt.ylabel('Accuracy', **axis_font)

        plt.xticks(size=font_size_axis)
        plt.yticks(size=font_size_axis)

        plt.legend(loc='upper left')
        plt.title(title, **title_font)
        plt.grid(True)

        fig1.savefig(output_history)

    def fit_model(self, x_train, y_train, x_validation=None, y_validation=None, class_weights=None):

        # -- configure the GPU that will be used
        self.set_gpu_configuration()

        # -- define the architecture
        self.architecture_definition()

        # -- choose the optimizer that will be used during the training process
        optimizer_methods = {'SGD': keras.optimizers.SGD,
                             'Adam': keras.optimizers.Adam,
                             'Adagrad': keras.optimizers.Adagrad,
                             'Adadelta': keras.optimizers.Adadelta,
                             }

        try:
            opt = optimizer_methods[self.optimizer]
        except KeyError:
            raise Exception('The optimizer %s is not being considered in this work yet:' % self.optimizer)

        # --  configure the learning process
        self.model.compile(loss=self.loss_function, optimizer=opt(lr=self.lr, decay=self.decay), metrics=['accuracy'])

        # -- define the callbacks
        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=20, verbose=0, mode='auto'),
                     ]

        # -- fit a model
        history = self.model.fit(x_train, y_train,
                                 batch_size=self.batch_size, epochs=self.epochs, verbose=1,
                                 callbacks=callbacks,
                                 validation_data=(x_validation, y_validation),
                                 shuffle=True,
                                 class_weight=class_weights,
                                 )

        # -- save the training history
        self.saving_training_history(history)

    def training(self, x_train, y_train, x_validation=None, y_validation=None):
        """ This method implements the training process of our CNN.

        Args:
            x_train (numpy.ndarray): Training data
            y_train (numpy.ndarray): Labels of the training data
            x_validation (:obj: `numpy.ndarray`, optional): Testing data. Defaults to None.
            y_validation (:obj: `numpy.ndarray`, optional): Labels of the testing data. Defaults to None.

        """

        if self.force_train or not os.path.exists(self.output_model):
            print('-- training ...')
            sys.stdout.flush()

            # -- compute the class weights for unbalanced datasets
            class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

            # -- convert class vectors to binary class matrices.
            y_train = keras.utils.to_categorical(y_train, self.num_classes)
            if y_validation is not None:
                y_validation = keras.utils.to_categorical(y_validation, self.num_classes)

            # -- fit the model
            self.fit_model(x_train, y_train, x_validation=x_validation, y_validation=y_validation, class_weights=class_weights)

            # -- save the fitted model
            print("-- saving model", self.output_model)
            sys.stdout.flush()

            self.model.save(self.output_model)
            self.model.save_weights(self.output_weights)
        else:
            print('-- model already exists in', self.output_model)
            sys.stdout.flush()

    def testing(self, x_test, y_test):
        """ This method is responsible for testing the fitted model.

        Args:
            x_test (numpy.ndarray): Testing data
            y_test (numpy.ndarray): Labels of the Testing data

        Returns:
            A dictionary with the ground-truth, the predicted scores and the predicted labels for the testing data.
            For example:
            {'gt': y_test,
             'predicted_labels': y_pred,
             'predicted_scores': y_scores,
            }

        """

        # -- configure the GPU that will be used
        self.set_gpu_configuration()

        # -- load the fitted model
        self.model = keras.models.load_model(self.output_model)

        # -- generates output predictions for the testing data.
        scores = self.model.predict(x_test, batch_size=self.batch_size, verbose=0)

        # -- get the predicted scores and labels for the testing data
        y_pred = np.argmax(scores, axis=1)
        y_scores = scores[:, 1]

        # -- define the output dictionary
        r_dict = {'gt': y_test,
                  'predicted_labels': y_pred,
                  'predicted_scores': y_scores,
                  }

        return r_dict

    def predict(self, x_values):
        """
        This method performs predictions using a previously trained model, returning the results.
        :param x_values: (numpy.ndarray)
        :return: (numpy.ndarray)
        """
        # -- configure the GPU that will be used
        self.set_gpu_configuration()

        # -- load the fitted model
        self.model = keras.models.load_model(self.output_model)

        # -- generates output predictions for the testing data.
        scores = self.model.predict(x_values, batch_size=self.batch_size, verbose=0)

        preds = np.argmax(scores, axis=1)

        return preds

    def run_evaluation_protocol(self):
        """ This method overrides the base classifier, because we have to transform images before feeding them to VGG. """

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
            'train_set': self.testing(dataset_protocol['train_set']['data'], dataset_protocol['train_set']['labels'])}

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
                              dtype=[('fname', 'U300'), ('pred_score', 'f8'), ('pred_labels', 'i8'), ('gt', 'i8')]
                              )
            with open(os.path.join(self.output_path, '%s.scores.txt' % key), 'w') as f:
                fw = csv.writer(f)
                fw.writerow(['fname', 'pred_score', 'pred_label', 'gt'])
                fw.writerows(output)
                # np.savetxt(os.path.join(self.output_path, '%s.scores.txt' % key),
                #            output,
                #            # header='fname;pred_score;pred_label',
                #            delimiter=',')

        # # -- create a mosaic for the positive and negative images.
        # all_pos_idxs = np.where(self.dataset.meta_info['all_labels'] == 1)[0]
        # all_neg_idxs = np.where(self.dataset.meta_info['all_labels'] == 0)[0]
        
        # # based on https://docs.python.org/3/library/itertools.html
        # def grouper(iterable, n, fillvalue=None):
        #     "Collect data into fixed-length chunks or blocks"
        #     # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
        #     args = [iter(iterable)] * n
        #     return list(zip_longest(*args, fillvalue=fillvalue))

        # for i, g in enumerate(grouper(self.dataset._imgs[all_pos_idxs, :, :, :], 50)):
        #     create_mosaic(g,
        #                 n_col=10,
        #                 output_fname=os.path.join(self.output_path, 'mosaic-pos-class-1_{}.jpeg'.format(i)))
        # for i, g in enumerate(grouper(self.dataset._imgs[all_neg_idxs, :, :, :], 50)):
        #     create_mosaic(g,
        #                 n_col=10,
        #                 output_fname=os.path.join(self.output_path, 'mosaic-neg-class-0_{}.jpeg'.format(i)))

        # -- saving the interesting samples for further analysis
        all_fnames = self.dataset.meta_info['all_fnames']
        self.interesting_samples(all_fnames, dataset_protocol['test_set'], class_report, predictions,
                                 threshold_type='EER')
