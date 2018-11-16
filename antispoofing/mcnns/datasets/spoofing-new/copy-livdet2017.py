import numpy as np
import os

# create a map of files
d = np.genfromtxt('filelisting.txt',delimiter=',',
    dtype=[('sequenceid','U15'),('path','U100')], skip_header=1)
pathmap = {f['sequenceid']: f['path'] for f in d}

# copy ndcld files to the appropriate folder
with open('livdet-sequenceids.txt','r') as f:
    files = [i.strip() for i in f.readlines() if 'sequenceid' not in i]

for f in files:
    src = pathmap[f]
    os.system('cp -v {} livdet2017/images'.format(src))
    # print(src)
