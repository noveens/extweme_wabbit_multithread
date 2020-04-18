#!/usr/bin/env bash

DATASET_NAME="warm_amz"
FILES_PREFIX="warm_amz"
BASE_PATH="/home/t-nosach/Data/Data/WarmAmazonTitles-300K/"
SED=sed

function xml_dataset4vw {
    FILE="$1"

    # Extract metadata
    INFO=$(head -n 1 $FILE | grep -o "[0-9]\+")
    INFOARRAY=($INFO)
    echo ${INFOARRAY[0]} >> ${FILE}.examples
    echo ${INFOARRAY[1]} >> ${FILE}.features
    echo ${INFOARRAY[2]} >> ${FILE}.labels

    echo "CONVERTING $FILE TO VW FORMAT ..."
    echo "${INFOARRAY[0]} EXAMPLES, ${INFOARRAY[1]} FEATURES, ${INFOARRAY[2]} LABELS"

    # Delete first line
    $SED -i "1d" $FILE

    # Add labels/features separator
    $SED -i "s/\(\(^\|,\| \)[0-9]\{1,\}\)  *\([0-9]\+:\)/\1 | \3/g" $FILE
    $SED -i "s/^ *\([0-9]\+:\)/\| \1/g" $FILE
}

# Copy dataset to required folder
if [ ! -e ./$FILES_PREFIX ]; then
    cp -R $BASE_PATH "./$FILES_PREFIX/"
fi

echo "PROCESSING ${FILES_PREFIX} ..."

cp  "./$FILES_PREFIX/train.txt"  "./$FILES_PREFIX/${FILES_PREFIX}_train"
xml_dataset4vw "./$FILES_PREFIX/${FILES_PREFIX}_train"

cp  "./$FILES_PREFIX/test.txt"  "./$FILES_PREFIX/${FILES_PREFIX}_test"
xml_dataset4vw "./$FILES_PREFIX/${FILES_PREFIX}_test"
