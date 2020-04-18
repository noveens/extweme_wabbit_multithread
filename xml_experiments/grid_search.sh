#!/usr/bin/env bash

DATASET_NAME="warm_amz"
FILES_PREFIX="warm_amz"
K=316053

# Clear log file
echo "" > grid_log.txt

for B in 32 33
do
    for kary in 32 64 128 256
    do
        echo "RUNNING FOR: B = ${B}, Kary = ${kary}" | tee -a grid_log.txt
        PARAMS="-l 0.0003 --power_t 0.2 --kary_tree ${kary} --passes 20 -b ${B}"
        bash run_xml.sh $DATASET_NAME $FILES_PREFIX $K "$PARAMS" 2>&1 | tee -a grid_log.txt
    done
done
