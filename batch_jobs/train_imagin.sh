IMAGIN_PATH=$PWD/imagin
ATLAS_PATH=$PWD/data
DATA_PATH=/scratch1/brizk/ADNI_SCREENING_DATA
DATA_TYPE=t1_linear
SPLITS_PATH=$DATA_PATH/splits
TARGET_DIR=$DATA_PATH/imagin_results

#w_s 5
BOLD_ARGS="--window_size 8 --window_stride 2"
SLICING_ARGS="--atlas_dir ${ATLAS_PATH} --data_type ${DATA_TYPE} --num_classes 3 --depth_of_slice 40 --slicing_stride 5"
PATHS_ARGS="--data_dir ${DATA_PATH} --splits_dir ${SPLITS_PATH} --targetdir ${TARGET_DIR}" 
TRAINING_ARGS1="--minibatch_size 16 --num_epochs 100 --clip_grad 0"
TRAINING_ARGS2="--lr 0.001 --max_lr 0.05 --reg_lambda 0.00001"
MODEL_ARGS="--hidden_dims 32 64 128 --sparsities 0 0 0 --dropout 0.1"

# NOTE INPUT_DIMS ARE 116 200 400

python $IMAGIN_PATH/main.py $PATHS_ARGS $BOLD_ARGS $SLICING_ARGS $TRAINING_ARGS1 $TRAINING_ARGS2 $MODEL_ARGS --no_kfold $1