# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse

from antispoofing.mcnns.utils import *
from antispoofing.mcnns.datasets import *
from antispoofing.mcnns.controller import *
from antispoofing.mcnns.classification import *


def call_controller(args):

    dataset = registered_datasets[args.dataset]

    data = dataset(args.dataset_path,
                   ground_truth_path=args.ground_truth_path,
                   permutation_path=args.permutation_path,
                   iris_location=args.iris_location,
                   output_path=args.output_path,
                   operation=args.operation,
                   max_axis=args.max_axis,
                   augmentation=args.augmentation,
                   transform_vgg= (args.ml_algo == 5),
                   )

    data.output_path = os.path.join(args.output_path,
                                    str(data.__class__.__name__).lower(),
                                    )

    control = Controller(data, args)
    control.execute_protocol()


def main():

    available_protocols = ["protocol_a"]

    dataset_options = 'Available dataset interfaces: '
    for k in sorted(registered_datasets.keys()):
        dataset_options += ('%s-%s, ' % (k, registered_datasets[k].__name__))

    ml_algo_options = 'Available Algorithm for Classification: '
    for k in sorted(ml_algo.keys()):
        ml_algo_options += ('%s-%s, ' % (k, ml_algo[k].__name__))

    losses_functions_options = 'Available Algorithm for Losses: '
    for k in sorted(loss_functions.keys()):
        losses_functions_options += ('%s-%s, ' % (k, loss_functions[k]))

    optimizer_methods_options = 'Available Optimizers: '
    for k in sorted(optimizer_methods.keys()):
        optimizer_methods_options += ('%s-%s, ' % (k, optimizer_methods[k]))

    # -- define the arguments available in the command line execution
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # -- arguments related to the dataset and to the output
    group_a = parser.add_argument_group('Arguments')

    group_a.add_argument('--dataset', type=int, metavar='', default=0, choices=range(len(registered_datasets)),
                         help=dataset_options + '(default=%(default)s).')

    group_a.add_argument('--dataset_path', type=str, metavar='', default='',
                         help='Path to the dataset.')

    group_a.add_argument('--output_path', type=str, metavar='', default='working',
                         help='Path where the results will be saved (default=%(default)s).')

    group_a.add_argument('--ground_truth_path', type=str, metavar='', default='',
                         help='A .csv file containing the ground-truth (default=%(default)s).')

    group_a.add_argument('--permutation_path', type=str, metavar='', default='',
                         help='A .csv file containing the data divided in two sets, the training and testing sets (default=%(default)s).')

    group_a.add_argument('--iris_location', type=str, metavar='', default='extra/irislocation_osiris.csv',
                         help='A .csv file containing the irises locations (default=%(default)s).')

    group_a.add_argument('--augmentation', type=int, default=0,
                         help='Apply augmentation to training data.')

    # -- arguments related to the Feature extraction module
    group_b = parser.add_argument_group('Available Parameters for Feature Extraction')

    group_b.add_argument('--feature_extraction', action='store_true',
                         help='Execute the feature extraction process (default=%(default)s).')

    group_b.add_argument("--descriptor", type=str, default="RawImage", metavar="",
                         choices=['RawImage', 'bsif'],
                         help="Allowed values are: " + ", ".join(available_protocols) +
                              " (default=%(default)s)")
    group_b.add_argument("--desc_params", type=str, default="", metavar="",
                         help="Additional parameters for feature extraction.")

    # -- arguments related to the Classification module
    group_c = parser.add_argument_group('Available Parameters for Classification')

    group_c.add_argument('--classification', action='store_true',
                         help='Execute the classification process (default=%(default)s).')

    group_c.add_argument('--ml_algo', type=int, metavar='', default=0, choices=range(len(ml_algo)),
                         help=ml_algo_options + '(default=%(default)s).')

    group_c.add_argument('--epochs', type=int, metavar='', default=300,
                         help='Number of the epochs considered during the learning stage (default=%(default)s).')

    group_c.add_argument('--bs', type=int, metavar='', default=32,
                         help='The size of the batches (default=%(default)s).')

    group_c.add_argument('--lr', type=float, metavar='', default=0.01,
                         help='The learning rate considered during the learning stage (default=%(default)s).')

    group_c.add_argument('--decay', type=float, metavar='', default=0.0,
                         help='The decay value considered during the learning stage (default=%(default)s).')

    group_c.add_argument('--reg', type=float, metavar='', default=0.1,
                         help='The value of the L2 regularization method (default=%(default)s).')

    group_c.add_argument('--loss_function', type=int, metavar='', default=0, choices=range(len(losses_functions_options)),
                         help=losses_functions_options + '(default=%(default)s).')

    group_c.add_argument('--optimizer', type=int, metavar='', default=0, choices=range(len(optimizer_methods)),
                         help=optimizer_methods_options + '(default=%(default)s).')

    group_c.add_argument('--fold', type=int, metavar='', default=0,
                         help='(default=%(default)s).')

    group_c.add_argument('--force_train', action='store_true',
                         help='(default=%(default)s).')

    group_c.add_argument('--cnn_ksize', type=int, metavar='', default=0,
                         help='(default=%(default)s).')

    group_d = parser.add_argument_group('Other options')

    group_d.add_argument('--operation', type=str, metavar='', default='crop', choices=['none', 'crop', 'segment'],
                         help='(default=%(default)s).')

    group_d.add_argument('--max_axis', type=int, metavar='', default=260,
                         help='(default=%(default)s).')

    group_d.add_argument('--device_number', type=str, metavar='', default='0',
                         help='(default=%(default)s).')

    parser.add_argument('--n_jobs', type=int, metavar='int', default=N_JOBS,
                        help='Number of jobs to be used during processing (default=%(default)s).')

    deprecated = parser.add_argument_group('Deprecated arguments')

    deprecated.add_argument('--last_layer', type=str, metavar='', default='linear', choices=['linear', 'softmax'],
                         help='(default=%(default)s).')

    deprecated.add_argument('--layers_name', nargs='+', type=str, metavar='', default=['conv_1'],
                         help='(default=%(default)s).')

    deprecated.add_argument('--fv', action='store_true',
                         help='(default=%(default)s).')

    args = parser.parse_args()

    print('ARGS:', args)
    sys.stdout.flush()

    call_controller(args)


if __name__ == "__main__":
    start = time.time()

    main()

    elapsed = (time.time() - start)
    print('Total time elaposed: {0}!'.format(time.strftime("%d days, and %Hh:%Mm:%Ss", time.gmtime(elapsed))))
    sys.stdout.flush()
