#!/bin/csh
module load python/3.5.2
source ~/pve/p35ocv/bin/activate.csh
python test.py matlab/UniversityOfOulu.jpg
# mv output/module.png output/module_py.png
