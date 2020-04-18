#!/usr/bin/env bash

DATASET_NAME="warm_amz"
FILES_PREFIX="warm_amz"
K=316053
PARAMS="-l 0.0003 --power_t 0.2 --kary_tree 64 --passes 20 -b 33"

bash run_xml.sh $DATASET_NAME $FILES_PREFIX $K "$PARAMS"
