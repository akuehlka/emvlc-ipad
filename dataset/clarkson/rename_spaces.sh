#/bin/bash

SAVEIFS=$IFS
IFS=$(echo -en "\n\b")
for f in `find ./images/ -type f -name "*.JPG" | grep " "`; do
    echo "$f -> ${f/ /}"
    mv $f ${f/ /}
done
IFS=$SAVEIFS