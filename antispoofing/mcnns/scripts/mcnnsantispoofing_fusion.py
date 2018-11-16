# -*- coding: utf-8 -*-

import sys
import time
import argparse
import json
import pdb
from antispoofing.mcnns.controller import *


def call_controller(args):

    control = FusionController(args)
    control.execute_protocol()


def main():

    # -- define the arguments available in the command line execution
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # -- arguments related to the dataset and to the output
    group_a = parser.add_argument_group('Arguments')

    group_a.add_argument('config', type=str,
                         help='JSON configuration file.')

    group_a.add_argument('--device_number', type=str, metavar='', default='0',
                         help='GPU device.')

    group_a.add_argument('--permutation', type=int, default=-1,
                         help='Permutation count.')

    group_a.add_argument('--featsel', type=str, default='',
                         help='Type of feature selection (fw, mi, rnd, or man, currently).')

    group_a.add_argument('--nfeats', type=int, default=0,
                         help='Number of features to be used.')

    group_a.add_argument('--features', type=str, default='',
                         help='Sequence of features to be used (as a python list).')

    group_a.add_argument('--weighttype', type=str, default='',
                         help='Type of weight used: imp or acc')

    group_a.add_argument('--augmentation', type=int, default=0,
                         help='Apply augmentation to training data.')

    group_a.add_argument('--force_predict', action='store_true',
                         help='(default=%(default)s).')

    c_arg = parser.parse_args()

    print('ARGS:', c_arg)

    with open(c_arg.config) as jdf:
        args = json.load(jdf)

    # copy dynamic parameters from the 'regular' arguments
    args['device_number'] = c_arg.device_number
    if c_arg.permutation > -1:
        args['permutation_path'] = args['permutation_path'].format(perm=c_arg.permutation)
    else:
        args['permutation_path'] = ''
    for cl in args['pre-classifiers']['classifier_list']:
        try:
            cl['permutation_path'] = cl['permutation_path'].format(perm=c_arg.permutation)
        except KeyError:
            cl['permutation_path'] = ''

    if c_arg.featsel:
        args['featsel'] = c_arg.featsel

    if c_arg.nfeats:
        if c_arg.nfeats > 0:
            args['nfeats'] = c_arg.nfeats

    if c_arg.features:
        args['features'] = eval(c_arg.features)

    if c_arg.weighttype:
        args['weighttype'] = c_arg.weighttype

    args['force_predict'] = c_arg.force_predict

    args['augmentation'] = bool(c_arg.augmentation)

    print(args)
    sys.stdout.flush()

    call_controller(args)


if __name__ == "__main__":
    start = time.time()

    main()

    elapsed = (time.time() - start)
    print('Total time elaposed: {0}!'.format(time.strftime("%d days, and %Hh:%Mm:%Ss", time.gmtime(elapsed))))
    sys.stdout.flush()
