IMAGIN_PATH=$PWD/imagin

DATA_DIR_PREFIX=/scratch1/${USER}/alz

#w_s 5
SLICING_ARGS="--num_classes 3"
BOLD_ARGS="--window_size 6 --window_stride 2"

TRAINING_ARGS1="--num_epochs 400 --clip_grad 2 --minibatch_size 8"
TRAINING_ARGS2="--lr 0.0001 --max_lr 0.01 --reg_lambda 0.0001 --label_smoothing 0.5"

MODEL_ARGS="--hidden_dims 32 64 64 --dropout 0.2"
# MODEL_ARGS="--hidden_dims 64 128 264 --sparsities 0 0 0 --dropout 0.1"

# NOTE INPUT_DIMS ARE 116 200 400
python $IMAGIN_PATH/main.py $SLICING_ARGS $BOLD_ARGS $TRAINING_ARGS1 $TRAINING_ARGS2 $MODEL_ARGS \
    --no_kfold --k_fold 1 \
    --caps_dir $DATA_DIR_PREFIX/ALL_ADNI_MRI_T1/ADNI_CAPS \
    --caps_type t1_volume \
    --atlases_dir $DATA_DIR_PREFIX/atlases \
    --train_split_filepath $DATA_DIR_PREFIX/ALL_ADNI_MRI_T1/ADNI_SPLITS/train.tsv \
    --val_split_filepath $DATA_DIR_PREFIX/ALL_ADNI_MRI_T1/ADNI_SPLITS/validationbaseline.tsv \
    --test_split_filepath $DATA_DIR_PREFIX/ALL_ADNI_MRI_T1/ADNI_SPLITS/testbaseline.tsv \
    --target_dir $DATA_DIR_PREFIX/ALL_ADNI_MRI_T1/imagin_results \
    $@