#!/bin/csh
module load python/3.5.2
setenv LD_LIBRARY_PATH .:$LD_LIBRARY_PATH
# ./test 04261d3665_imno.bmp
./test matlab/UniversityOfOulu.jpg
# mv output/module.png output/module_cpp.png
