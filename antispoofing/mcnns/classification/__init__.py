# -*- coding: utf-8 -*-

from antispoofing.mcnns.classification.cnn import CNN
from antispoofing.mcnns.classification.lrn import LRN2D
from antispoofing.mcnns.classification.simplefuser import SimpleFuser
from antispoofing.mcnns.classification.votingfuser import VotingFuser
from antispoofing.mcnns.classification.treefuser import TreeFuser
from antispoofing.mcnns.classification.weightedvotingfuser import WeightedVotingFuser
from antispoofing.mcnns.classification.vgg import VGG
from antispoofing.mcnns.classification.lenet import LeNet
from antispoofing.mcnns.classification.alexnet import AlexNet
from antispoofing.mcnns.classification.vgg13 import VGG13
from antispoofing.mcnns.classification.linearsvm import LinearSVM


ml_algo = {0: CNN,
           1: SimpleFuser,
           2: VotingFuser,
           3: TreeFuser,
           4: WeightedVotingFuser,
           5: VGG,
           6: LeNet,
           7: AlexNet,
           8: VGG13,
           9: LinearSVM,
           }


loss_functions = {0: 'categorical_crossentropy',
                  1: 'sparse_categorical_crossentropy',
                  2: 'categorical_hinge',
                  3: 'hinge',
                  4: 'binary_crossentropy',
                  }


optimizer_methods = {0: 'SGD',
                     1: 'Adam',
                     2: 'Adagrad',
                     3: 'Adadelta',
                     4: 'Adamax',
                     }
