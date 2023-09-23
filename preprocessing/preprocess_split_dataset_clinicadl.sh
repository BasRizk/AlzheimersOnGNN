PREFIX_DATA_DIR=/scratch1/$USER/alz
BIDS_DIR=$PREFIX_DATA_DIR/ALL_ADNI_MRI_T1/ADNI_BIDS
OUTPUT_DIR=$PREFIX_DATA_DIR/ALL_ADNI_MRI_T1/ADNI_SPLITS

echo "getting labels"
clinicadl tsvtools get-labels $BIDS_DIR $OUTPUT_DIR --diagnoses AD --diagnoses CN --diagnoses MCI --diagnoses EMCI --diagnoses SMC
echo "finished getting labels"

echo "splitting labels"
clinicadl tsvtools split $OUTPUT_DIR/labels.tsv --n_test 0.2 --subset_name test 
echo "finished splitting labels"

echo "splitting labels"
clinicadl tsvtools split $OUTPUT_DIR/split/train.tsv --subset_name validation 
echo "finished splitting labels"