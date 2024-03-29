DATA_DIR=/scratch1/$USER/alz
# RAW_DATASET_DIRECTORY=$DATA_DIR/ADNI
RAW_DATASET_DIRECTORY=$DATA_DIR/ALL_ADNI_MRI_T1/ADNI
CLINICAL_DATA_DIRECTORY=$DATA_DIR/clinical_data

BIDS_DIRECTORY=${RAW_DATASET_DIRECTORY}_BIDS
# Extract only T1 modality from ADNI dataset
OPTIONS="-m T1"

clinica -v convert adni-to-bids $OPTIONS $RAW_DATASET_DIRECTORY $CLINICAL_DATA_DIRECTORY $BIDS_DIRECTORY 