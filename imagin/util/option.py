import os
import argparse

def parse():
    parser = argparse.ArgumentParser(description='SPATIO-TEMPORAL-ATTENTION-GRAPH-ISOMORPHISM-NETWORK')
    
    parser.add_argument('--label_smoothing', type=float, default=0.0)

    parser.add_argument('--use_cached', action='store_true', help='for some debugging behavior')
    parser.add_argument('--clean_ckpt', action='store_true')
    parser.add_argument('--no_kfold', action='store_true')

    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--depth_of_slice', type=int, default=40)
    parser.add_argument('--slicing_stride', type=int, default=5)
    parser.add_argument('--preserve_img_shape_in_slices', type=bool, default=True)
    parser.add_argument(
        '--parallize_brains', action='store_true', default=False,
        help="parallize per brains processing instead of"
         "parallizing threads over slices processing"
         " (set to true when working with real dataset)"
    )

    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-n', '--exp_name', type=str, default='adni_exp')
    parser.add_argument('-k', '--k_fold', type=int, default=1) #5
    parser.add_argument('-b', '--minibatch_size', type=int, default=32) 

    username = os.environ['USER']
    data_dir_prefix = f'/scratch1/{username}/alz'
    caps_dir=f'{data_dir_prefix}/ALL_ADNI_MRI_T1/ADNI_CAPS'
    caps_type='t1_volume'
    atlases_dir=f'{data_dir_prefix}/atlases'
    train_split_filepath=f'{data_dir_prefix}/ALL_ADNI_MRI_T1/ADNI_SPLITS/train.tsv'
    val_split_filepath=f'{data_dir_prefix}/ALL_ADNI_MRI_T1/ADNI_SPLITS/validationbaseline.tsv'
    test_split_filepath=f'{data_dir_prefix}/ALL_ADNI_MRI_T1/ADNI_SPLITS/testbaseline.tsv'


    parser.add_argument(
        '--caps_type', type=str, default=caps_type, choices=['t1_linear', 't1_volume'],
        help='whether the data are processed using t1_linear or t1_volume pipeline'
    )
    parser.add_argument('--caps_dir', type=str, default=caps_dir)
    parser.add_argument('--atlases_dir', type=str, default=atlases_dir)
    parser.add_argument('--train_split_filepath', type=str, default=train_split_filepath)
    parser.add_argument('--val_split_filepath', type=str, default=val_split_filepath)
    parser.add_argument('--test_split_filepath', type=str, default=test_split_filepath)
    parser.add_argument('--target_dir', type=str, default='./imagin_results')
   
    # TODO consider making the window size and stride dynamic given percentage and the depth of slice!!
    parser.add_argument('--window_size', type=int, default=5) #50
    parser.add_argument('--window_stride', type=int, default=1) #3
    parser.add_argument('--dynamic_length', type=int, default=None) #600

    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--max_lr', type=float, default=0.001)
    parser.add_argument('--reg_lambda', type=float, default=0.00001)
    parser.add_argument('--clip_grad', type=float, default=0.0)
    parser.add_argument('--num_epochs', type=int, default=30)
    # NOT USED
    # parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--num_layers', nargs="+", type=int, default=[4, 4, 4])
    parser.add_argument('--hidden_dims', nargs="+", type=int, default=[128, 128,128])
    parser.add_argument('--sparsities', nargs="+", type=int, default=[0, 0, 0]) #[30, 30, 30] # TODO maybe come back to later
    parser.add_argument('--dropout', type=float, default=0.5)
    # NOT USED
    # parser.add_argument('--readout', type=str, default='sero', choices=['garo', 'sero', 'mean']) 
    parser.add_argument('--cls_token', type=str, default='sum', choices=['sum', 'mean', 'param'])
    parser.add_argument('--no_train', action='store_true')
    parser.add_argument('--no_test', action='store_true')
    parser.add_argument('--no_analysis', action='store_true')

    argv = parser.parse_args()
    
    return argv
