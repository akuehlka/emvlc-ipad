import argparse
import csv
import os

from antispoofing.mcnns.utils.classifierselector import ClassifierSelector

testgroups = ['train_set', 'val_set', 'test', 'unknown_test']

ap = argparse.ArgumentParser()
ap.add_argument('path', type=str, default='output')
ap.add_argument('outputfile', type=str, default='protocol_summary')
ap.add_argument('--descriptor', type=str, default='')
ap.add_argument('--dsname', type=str, default='')
args = ap.parse_args()

CS = ClassifierSelector(args.path, [args.dsname], [args.descriptor], testgroups)
# CS = ClassifierSelector('output', ['livdetiris17_nd','livdetiris17_warsaw'], ['RawImage','bsif'], testgroups)

summary = CS.summarize_results_new()

fname = os.path.join(args.path,args.outputfile)
if args.dsname:
    fname += '_' + args.dsname
if args.descriptor:
    fname += '_' + args.descriptor
fname += '.csv'

with open(fname, 'w') as fout:
    w = csv.writer(fout)
    w.writerow(summary.dtype.names)
    w.writerows(summary)

# CS.select_classifiers()