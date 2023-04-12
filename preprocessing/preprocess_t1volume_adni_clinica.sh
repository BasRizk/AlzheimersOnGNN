. ./preprocessing/source_env.sh

NUM_PROCS=1
# TMP_WORK_DIR=WD_subset
# ADNI_CONVERTED_DIR="data/ADNI_MRI_Source_rs-fMRI_BIDS"
ADNI_CONVERTED_DIR='data/SUBSET_OF_DATA'
CLINICAL_DATA_DIRECTORY="data/clinical_data_merged"
PROCESSED_DIR="${ADNI_CONVERTED_DIR}_PROCESSED"
# TSV_NAME="AD"
# TSV_DIR="data/ADNIS_Splits/${TSV_NAME}.tsv"

# OPTIONS="-tsv $TSV_DIR -wd './WD_${TSV_NAME}' -np 24"

clinica run t1-volume $ADNI_CONVERTED_DIR $PROCESSED_DIR 'TRAIN' -np $NUM_PROCS