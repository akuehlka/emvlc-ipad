#!/bin/bash
SRC=$1
if [ -z "$1" ]; then
    SRC="output"
fi
DST=$2
if [ -z "$2" ]; then
    DST="protocol_summary"
fi

OS=`uname -s`
if [ "${OS}" = "Darwin" ]; then
    source activate keras_anaconda
    cd ~/src/iris-spoofing-detection
else
    module purge
    module load python/3.6.4
    # -- activate the python virtual environment
    source ~/pve/p36ocv/bin/activate
    cd ~/src/isd/
fi

export PYTHONPATH="."

python scripts/summarize_accuracy.py ${SRC} ${DST}