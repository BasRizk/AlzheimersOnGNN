source preprocessing/source_env.sh

# $1 is the first argument (PREFIX_DATA_DIR)
PREFIX_DATA_DIR=$1
# $2 is the second argument (BIDS_DIR)
BIDS_DIR=$2
# $3 is the third argument (GROUP_LABEL)
GROUP_LABEL=$3
# $4 is the fourth argument (CAPS_DIR)
CAPS_DIR=$4
# ${@:5} represents all arguments starting from the fifth one (TSV_FILES)
ALL_TSV_FILES=${@:5}
# convert ALL_TSV_FILES to an array
ALL_TSV_FILES=($ALL_TSV_FILES)

echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
TSV_FILE=${ALL_TSV_FILES[$SLURM_ARRAY_TASK_ID]}

# Now you can use these variables in your script
echo "PREFIX_DATA_DIR: $PREFIX_DATA_DIR"
echo "BIDS_DIR: $BIDS_DIR"
echo "GROUP_LABEL: $GROUP_LABEL"
echo "CAPS_DIR: $CAPS_DIR"
echo "TSV_FILE: $TSV_FILE"

NUM_OF_SAMPLES=$(wc -l < $TSV_FILE)
TMP_WORK_DIR=$PREFIX_DATA_DIR/ADNI_SPLITS/split/WD_${GROUP_LABEL}_${SLURM_ARRAY_TASK_ID}
OPTIONS="--n_procs 1 -tsv $TSV_FILE -wd $TMP_WORK_DIR"

echo Running t1-volume pipeline on $TSV_FILE: $subject
echo "Running t1-volume pipeline on ${GROUP_LABEL} group"
echo "Number of samples in $TSV_FILE: $NUM_OF_SAMPLES"
echo "Temporary working directory: $TMP_WORK_DIR"
echo "Options: $OPTIONS"


# extract TSV_FILE directory
TSV_DIR=$(dirname $TSV_FILE)
# extract TSV_FILE basename
TSV_BASENAME=$(basename $TSV_FILE)
# extract TSV_FILE basename without extension
TSV_BASENAME_NO_EXT=${TSV_BASENAME%.*}

# make directory for logs of this TSV_FILE
mkdir -p $TSV_DIR/logs
mkdir -p $TSV_DIR/logs/$TSV_BASENAME_NO_EXT
cd $TSV_DIR/logs/$TSV_BASENAME_NO_EXT

# make directory for logs of this TSV_FILE
mkdir -p $TSV_DIR/logs
mkdir -p $TSV_DIR/logs/$TSV_BASENAME_NO_EXT
cd $TSV_DIR/logs/$TSV_BASENAME_NO_EXT

# Note train here is used because it is the name of the existing template
clinica run t1-volume-existing-template $OPTIONS $BIDS_DIR $CAPS_DIR "train"

echo "Finished running t1-volume pipeline on ${GROUP_LABEL} group from $TSV_FILE"


