# -*- coding: utf-8 -*-

import os
import sys
import json

from antispoofing.mcnns.utils.common_imports import *
from antispoofing.mcnns.classification.baseclassifier import BaseClassifier
from sklearn.utils import class_weight
# from vis.utils import utils
# from vis.visualization import visualize_activation
# from vis.visualization import visualize_saliency


class VGG13(BaseClassifier):
    def __init__(self, output_path, dataset, input_shape=200, epochs=50,
                 batch_size=8, loss_function='categorical_crossentropy', lr=0.01, decay=0.0005, optimizer='SGD', regularization=0.1,
                 device_number=0, force_train=False, filter_vis=False, layers_name=('conv_1',),
                 ksize=(3,3),
                 fold=0):

        super(VGG13, self).__init__(output_path, dataset,
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

        img_input = Input(shape=self.input_shape, name='input_1')

        # -- first layer
        conv2d_1 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='conv_1')(img_input)
        conv2d_2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu', name='conv_2')(conv2d_1)
        max_pooling_1 = MaxPooling2D(pool_size=(2,2), name='pool_1')(conv2d_2)
        conv2d_3 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='conv_3')(max_pooling_1)
        conv2d_4 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='conv_4')(conv2d_3)
        max_pooling_2 = MaxPooling2D(pool_size=(2,2), name='pool_2')(conv2d_4)
        conv2d_5 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='conv_5')(max_pooling_2)
        conv2d_6 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='conv_6')(conv2d_5)
        max_pooling_3 = MaxPooling2D(pool_size=(2,2), name='pool_3')(conv2d_6)
        conv2d_7 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='conv_7')(max_pooling_3)
        conv2d_8 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='conv_8')(conv2d_7)
        max_pooling_4 = MaxPooling2D(pool_size=(2,2), name='pool_4')(conv2d_8)
        conv2d_9 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='conv_9')(max_pooling_4)
        conv2d_10 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='conv_10')(conv2d_9)
        max_pooling_5 = MaxPooling2D(pool_size=(2,2), name='pool_5')(conv2d_10)
        
        # -- fully-connected layer
        flatten_1 = Flatten(name='flatten')(max_pooling_5)
        dense_1 = Dense(units=4096, activation='relu', name='fc1')(flatten_1)
        dropout_1 = Dropout(0.5)(dense_1)
        dense_2 = Dense(units=4096, activation='relu', name='fc2')(dropout_1)
        dropout_2 = Dropout(0.5)(dense_2)

        # -- classification block
        output = Dense(self.num_classes, activation='softmax', name='predictions',
                       kernel_regularizer=keras.regularizers.l2(self.regularization))(dropout_2)

        self.model = keras.models.Model(img_input, output, name='lenet')

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