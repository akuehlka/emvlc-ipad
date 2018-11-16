
from antispoofing.mcnns.utils.constants import SEED

import tensorflow as tf
tf.set_random_seed(SEED)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab

import keras
from keras import activations
from keras import backend as K
from keras.models import Sequential
from keras.layers import Input
from keras.layers import InputLayer
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import LocallyConnected2D
from keras.layers import advanced_activations
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Layer
from keras.layers import Dropout
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

