OPTIONS="-m T1"
DATASET_DIRECTORY="data/ADNI_MRI_Source_rs-fMRI"
CLINICAL_DATA_DIRECTORY="data/clinical_data_merged"
BIDS_DIRECTORY="data/${DATASET_DIRECTORY}_BIDS"

clinica -v convert adni-to-bids $DATASET_DIRECTORY $CLINICAL_DATA_DIRECTORY $BIDS_DIRECTORY $OPTIONS
# clinica -v convert adni-to-bids $DATASET_DIRECTORY $CLINICAL_DATA_DIRECTORY $BIDS_DIRECTORY
