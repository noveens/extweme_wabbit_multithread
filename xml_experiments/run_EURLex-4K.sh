#!/usr/bin/env bash

DATASET_NAME="EURLex-4K"
FILES_PREFIX="eurlex"
K=3993
PARAMS="-l 0.0003 --power_t 0.2 --kary_tree 64 --passes 1 -b 10"

bash run_xml.sh $DATASET_NAME $FILES_PREFIX $K "$PARAMS"
