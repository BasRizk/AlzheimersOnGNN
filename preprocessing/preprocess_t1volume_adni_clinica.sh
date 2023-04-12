. ./preprocessing/source_env.sh

NUM_PROCS=6
TMP_WORK_DIR= 'WD_train'
ADNI_CONVERTED_DIR="data/ADNI_MRI_Source_rs-fMRI_BIDS"
CLINICAL_DATA_DIRECTORY="data/clinical_data_merged"
PROCESSED_DIR="${ADNI_CONVERTED_DIR}_PROCESSED"
# TSV_NAME="AD"
# TSV_DIR="data/ADNIS_Splits/${TSV_NAME}.tsv"

# OPTIONS="-tsv $TSV_DIR -wd './WD_${TSV_NAME}' -np 24"

clinica run t1-volume $ADNI_CONVERTED_DIR $PROCESSED_DIR 'TRAIN' -wd ./${TMP_WORK_DIR} -np $NUM_PROCS