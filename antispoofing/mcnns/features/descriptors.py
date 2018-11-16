# -*- coding: utf-8 -*-

import numpy as np
from skimage import feature
from antispoofing.mcnns.utils import *
from skimage.feature import greycomatrix, greycoprops
from skimage.util.shape import view_as_windows
# from pywt import WaveletPacket2D


from scipy import ndimage as ndi
from skimage.util import img_as_float
from skimage.filters import gabor_kernel

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

# required for BSIF
import cv2
import bsif


class RawImage(object):

    def __init__(self):
        pass

    def extraction(self, img):
        return np.reshape(img, (1, -1))


class BSIF(object):

    def __init__(self, filter_dimensions):
        self.filter_dimensions = filter_dimensions

    def extraction(self, img):
        bsif_im = np.zeros_like(img)
        bsif_im = bsif.extract(img, bsif_im, self.filter_dimensions)

        # convert and scale the image to uint8
        scale = bsif_im/(2**self.filter_dimensions[2])
        bsif_im = scale*255

        # pdb.set_trace()
        # cv2.imwrite('bsifim.png', bsif_im.astype(np.uint8))

        return bsif_im.astype(np.uint8)
