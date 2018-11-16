#!/bin/bash

FULL_IMAGE_LIST=$1
OSIRIS=/afs/crc.nd.edu/user/a/akuehlka/src/Iris_Osiris_v4.1/src/osiris
PARAMSPATH=/afs/crc.nd.edu/user/a/akuehlka/src/isd/dataset/iiitd/osiris

rm tasklist
rm tmp/*

# split the images in chunks of 3000
split -d -l 200 $FULL_IMAGE_LIST tmp/images

# each of the chunks will start a process
for l in `ls tmp/images?? | cut -d / -f 2`; do
    IMAGE_LIST="tmp/$l"
    echo $IMAGE_LIST

    # create directories for output
    NORM_OUTPUT="osiris_output/"
    mkdir -p $NORM_OUTPUT

    # prepare the config file
    CONFIG_NAME="tmp/normalize_$l.ini"
    sed "s/@1@/${IMAGE_LIST//\//\\/}/g" "${PARAMSPATH}/normalize.ini" > $CONFIG_NAME
    sed -i.bak -e "s/@2@/${NORM_OUTPUT//\//\\/}/g" $CONFIG_NAME
    sed -i.bak "s/@3@/${NORM_OUTPUT//\//\\/}/g" $CONFIG_NAME
    sed -i.bak "s/@4@/${NORM_OUTPUT//\//\\/}/g" $CONFIG_NAME
    
    # enqueue all commands
    echo "$OSIRIS $CONFIG_NAME 2>&1 > tmp/osiris_$l.log" >> tasklist
    
    # break
done

parallel --eta < tasklist

# remove the temporary lists when done
#rm tmp/images??
