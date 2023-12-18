import os
import time
import argparse
from loguru import logger
from dataset import ADNI

# set logging to trace for debugging
# logger.remove()
# logger.add(lambda msg: print(msg), level="TRACE")


parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=3)
parser.add_argument('--depth_of_slice', type=str, nargs='+', default=['40'])
parser.add_argument('--slicing_stride', type=str, nargs='+', default=['5'])
parser.add_argument('--n_processes', type=int, default=None)
parser.add_argument(
    '--overwrite_cache', action='store_true',
    help='If set, will overwrite the cache file if it exists'
)
parser.add_argument('--mode', type=str, nargs='+', default=['train', 'val', 'test'])
args = parser.parse_args()

username = os.environ['USER']
data_dir_prefix = f'/scratch1/{username}/alz'
caps_dir=f'{data_dir_prefix}/ALL_ADNI_MRI_T1/ADNI_CAPS'
caps_type='t1_volume'
atlases_dir=f'{data_dir_prefix}/atlases'

_split_filepathes={
    'train': f'{data_dir_prefix}/ALL_ADNI_MRI_T1/ADNI_SPLITS/train.tsv',
    'val' : f'{data_dir_prefix}/ALL_ADNI_MRI_T1/ADNI_SPLITS/validationbaseline.tsv',
    'test': f'{data_dir_prefix}/ALL_ADNI_MRI_T1/ADNI_SPLITS/testbaseline.tsv'
}

for slicing_stride in args.slicing_stride:
    slicing_stride = int(slicing_stride)
    logger.debug(f'slicing_stride = {slicing_stride}')
    if slicing_stride < 1:
        raise ValueError(f'Invalid slicing_stride {slicing_stride}. slicing_stride must be >= 1')
    
    for depth_of_slice in args.depth_of_slice:
        depth_of_slice = int(depth_of_slice)
        logger.debug(f'depth_of_slice = {depth_of_slice}')
        if depth_of_slice < 1:
            raise ValueError(f'Invalid depth_of_slice {depth_of_slice}. depth_of_slice must be >= 1')
        
        for mode in args.mode:
            logger.debug(f'mode = {mode}')
            if mode not in _split_filepathes:
                raise ValueError(f'Invalid mode {mode}. Valid modes are: {_split_filepathes.keys()}')
            
            split_filepath = _split_filepathes[mode]
            num_classes = args.num_classes
            

            logger.info(
                f'Preparing dataset with {num_classes} classes, '
                f'{depth_of_slice} depth_of_slice, '
                f'{slicing_stride} slicing_stride, '
                f'n_processes={args.n_processes}, '
                f'overwrite_cache={args.overwrite_cache}'
                f'mode = {args.mode}'
            )

            start_time = time.time()
            dataset: ADNI = ADNI(
                caps_dir,
                caps_type,
                atlases_dir,
                split_filepath,
                num_classes,
                depth_of_slice,
                slicing_stride,
                preserve_img_shape_in_slices=True,
                k_fold=1,
                parallize_brains_not_slices=True,
                n_processes=args.n_processes,
                overwrite_cache=args.overwrite_cache
            )
            logger.success(f'Finished preprocessing feats at {dataset.experiment_dir} of {len(dataset)} brains in {time.time() - start_time:.2f} seconds')
            logger.debug(f'Finished preprocessing of feats mode {args.mode} with args {args}')
            
        logger.success(f'Finished preprocessing feats per this call of prepare_feats.py')