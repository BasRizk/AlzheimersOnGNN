#!/bin/bash

# first go to data directory, grab all subjects,
# and assign to an array

PREFIX_DATA_DIR=/scratch1/$USER/alz/ALL_ADNI_MRI_T1
BIDS_DIR=$PREFIX_DATA_DIR/ADNI_BIDS
# set N_PROC to 32 if it is not passed as an argument
NPROC=${1:-61}

# TRAINING DATASET
GROUP_LABEL="train"
SPLITS_DIR=$PREFIX_DATA_DIR/ADNI_SPLITS/split/split
MAIN_TSV_FILE=$SPLITS_DIR/$GROUP_LABEL.tsv


# PROCESSED/CAPS_DIR
CAPS_DIR=$PREFIX_DATA_DIR/ADNI_CAPS_$GROUP_LABEL


NUM_OF_SAMPLES=$(wc -l < $MAIN_TSV_FILE)

echo "Number of samples in $MAIN_TSV_FILE: $NUM_OF_SAMPLES"

OUTPUT_DIR=$SPLITS_DIR/${GROUP_LABEL}_${NPROC}
rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR


# split the file into smaller files, including the header each
sh preprocessing/split_tsv_file.sh $MAIN_TSV_FILE $NPROC $OUTPUT_DIR
echo "Spawning $NPROC sub-jobs."

echo "Current directory = $PWD"

TSV_FILES=($(ls $OUTPUT_DIR/*))

sh preprocessing/verify_equal_header.sh ${TSV_FILES[@]}


len=$(expr ${#TSV_FILES[@]} - 1) # len - 1
echo "Number of files: ${#TSV_FILES[@]}"
# print list of files with their indices
for i in $(seq 0 $len); do
    echo "$i: ${TSV_FILES[$i]}"
done

sbatch --array=0-$len batch_jobs/processing_script.job preprocessing/preprocess_t1volume_adni_clinica.sh $PREFIX_DATA_DIR $BIDS_DIR $GROUP_LABEL $CAPS_DIR ${TSV_FILES[@]}