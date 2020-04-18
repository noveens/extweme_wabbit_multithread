#!/usr/bin/env bash

DATASET_NAME=$1
FILES_PREFIX=$2
K=$3
PARAMS=$4

SCRIPT_DIR=$( dirname "${BASH_SOURCE[0]}" )
BIN=${SCRIPT_DIR}/../vowpalwabbit/vw
TIMER=/usr/bin/time

# Clones repo with scripts to download datasets
if [ ! -e datasets4vw ]; then
   git clone https://github.com/mwydmuch/datasets4vw.git
fi

# Downloads dataset
if [ ! -e ${FILES_PREFIX}/${FILES_PREFIX}_train ]; then
    bash datasets4vw/xml_repo/get_${DATASET_NAME}.sh
fi

# Builds extweme(vowpal) wabbit
# if [ ! -e ${BIN} ]; then
#     cd ${SCRIPT_DIR}/..
#     make
#     cd -
# fi

TRAIN=${FILES_PREFIX}/${FILES_PREFIX}_train
TEST=${FILES_PREFIX}/${FILES_PREFIX}_test

if [ ! -e $TRAIN ]; then
    TRAIN=${FILES_PREFIX}/${FILES_PREFIX}_train0
    TEST=${FILES_PREFIX}/${FILES_PREFIX}_test0
fi

mkdir -p models
MODEL="models/${FILES_PREFIX}_$(echo $PARAMS | tr ' ' '_')"

# Trains model
#if [ ! -e $MODEL ]; then
    $TIMER $BIN --plt $K $TRAIN -f $MODEL --sgd $PARAMS -c
    
    #rm ${TRAIN}.c #remove cache file
#fi

# Tests model
THREADS=3
python test_multithread.py $MODEL $TEST ${FILES_PREFIX}/ $THREADS
# $TIMER $BIN -t -i  --top_k 5 -p preds_${FILES_PREFIX}
