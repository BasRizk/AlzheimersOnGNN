. ./preprocessing/source_env.sh

# Modify if necessary
NUM_PROCS=32
MODE='Train' # Test/Val/Train
PREFIX_DATA_DIR='/project/ajiteshs_1045/alzhiemers'
ADNI_CONVERTED_DIR='/scratch1/ag_394/ADNI3_MRI_BIDS'


CAPS_DIR='TRAIN'
TMP_WORK_DIR=./WD_${MODE}
CLINICAL_DATA_DIRECTORY="${PREFIX_DATA_DIR}/clinical_data_merged"
PROCESSED_DIR="${ADNI_CONVERTED_DIR}_PROCESSED"
TSV_DIR=splits/${MODE}_ADNI.tsv

clinica run t1-volume-existing-template $ADNI_CONVERTED_DIR $PROCESSED_DIR $CAPS_DIR -tsv $TSV_DIR -wd $TMP_WORK_DIR -np $NUM_PROCS
