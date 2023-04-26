IMAGIN_PATH=$PWD/imagin
ATLAS_PATH=$PWD/data
DATA_PATH=/scratch1/brizk/ADNI_SCREENING_DATA
DATA_TYPE=t1_linear
SPLITS_PATH=$DATA_PATH/splits
TARGET_DIR=$DATA_PATH/imagin_results

#w_s 5
BOLD_ARGS="--window_size 2 --window_stride 1"
SLICING_ARGS="--atlas_dir ${ATLAS_PATH} --data_type ${DATA_TYPE} --num_classes 3 --depth_of_slice 50 --slicing_stride 6"
PATHS_ARGS="--data_dir ${DATA_PATH} --splits_dir ${SPLITS_PATH} --targetdir ${TARGET_DIR}" 
MODEL_ARGS="--minibatch_size 1 --hidden_dims 32 64 128 --sparsities 15 15 15 --dropout 0.2"

# NOTE INPUT_DIMS ARE 116 200 400

python $IMAGIN_PATH/main.py $PATHS_ARGS $BOLD_ARGS $SLICING_ARGS $MODEL_ARGS --no_kfold $1