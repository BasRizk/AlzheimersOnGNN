import os
import csv
import argparse


def parse():
    parser = argparse.ArgumentParser(description='SPATIO-TEMPORAL-ATTENTION-GRAPH-ISOMORPHISM-NETWORK')
    
    parser.add_argument('--label_smoothing', type=float, default=0.0)

    parser.add_argument('--use_cached', action='store_true', help='for some debugging behavior')
    parser.add_argument('--clean_ckpt', action='store_true')
    parser.add_argument('--clean_cache', action='store_true')
    parser.add_argument('--no_kfold', action='store_true')

    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--depth_of_slice', type=int, default=40)
    parser.add_argument('--slicing_stride', type=int, default=5)
    parser.add_argument(
        '--parallize_brains', action='store_true', default=False,
        help="parallize per brains processing instead of"
         "parallizing threads over slices processing"
         " (set to true when working with real dataset)"
    )

    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-n', '--exp_name', type=str, default='adnimagin_experiment')
    parser.add_argument('-k', '--k_fold', type=int, default=2) #5
    parser.add_argument('-b', '--minibatch_size', type=int, default=2) #3

    data_path = atlas_path = '../data'
    data_type = 't1_volume'
    data_path = '/project/ajiteshs_1045/alzheimers/temp_t1linear_brains/'
    data_type = 't1_linear'
    split_path = f'../splits/splits_{data_type}_toy'
    parser.add_argument(
        '--data_type', type=str, default=data_type,
        help='whether the data are processed using t1_linear or t1_volume pipeline'
    )
    parser.add_argument('-ds', '--data_dir', type=str, default=data_path)
    parser.add_argument('-ad', '--atlas_dir', type=str, default=atlas_path)
    parser.add_argument('-sd', '--splits_dir', type=str, default=split_path)
    parser.add_argument('-dt', '--targetdir', type=str, default='./result')
    
    # parser.add_argument('--dataset', type=str, default='rest', choices=['rest', 'task'])
    # parser.add_argument('--roi', type=str, default='schaefer', choices=['scahefer', 'aal', 'destrieux', 'harvard_oxford'])
    # NOT USED
    parser.add_argument('--fwhm', type=float, default=None)

    parser.add_argument('--window_size', type=int, default=5) #50
    parser.add_argument('--window_stride', type=int, default=1) #3
    parser.add_argument('--dynamic_length', type=int, default=None) #600

    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--max_lr', type=float, default=0.001)
    parser.add_argument('--reg_lambda', type=float, default=0.00001)
    parser.add_argument('--clip_grad', type=float, default=0.0)
    parser.add_argument('--num_epochs', type=int, default=30)
    # NOT USED
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--num_layers', nargs="+", type=int, default=[4, 4, 4])
    parser.add_argument('--hidden_dims', nargs="+", type=int, default=[128, 128,128])
    parser.add_argument('--sparsities', nargs="+", type=int, default=[30, 30, 30])
    parser.add_argument('--dropout', type=float, default=0.5)
    # NOT USED
    parser.add_argument('--readout', type=str, default='sero', choices=['garo', 'sero', 'mean']) 
    parser.add_argument('--cls_token', type=str, default='sum', choices=['sum', 'mean', 'param'])

    # parser.add_argument('--num_clusters', type=int, default=7)
    # parser.add_argument('--subsample', type=int, default=50)

    parser.add_argument('--no_train', action='store_true')
    parser.add_argument('--no_test', action='store_true')
    parser.add_argument('--no_analysis', action='store_true')

    argv = parser.parse_args()
    argv.targetdir = os.path.join(argv.targetdir, argv.exp_name)
    os.makedirs(argv.targetdir, exist_ok=True)
    with open(os.path.join(argv.targetdir, 'argv.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(vars(argv).items())
    return argv
