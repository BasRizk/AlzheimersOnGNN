import os
import csv
import shutil
from analysis import analyze
from util.option import parse

def get_argv():    
    argv = parse()
    
    prefix = ""
    # if argv.use_cached:
    #     prefix = "cached_______"
        
    
    label_smoothing_suffix = f'ls({argv.label_smoothing})' if argv.label_smoothing > 0 else ''
    
    argv.target_dir = os.path.join(
        argv.target_dir,
        
        f'dslice{argv.depth_of_slice}'
        f'_stride{argv.slicing_stride}',
        
        f'{argv.exp_name}__'
        f'{prefix}ws{argv.window_size}'
        f'_st{argv.window_stride})'
        f'_lr{argv.lr}mx{argv.max_lr}'
        f'_reg{argv.reg_lambda}'
        f'{label_smoothing_suffix}',
        f'b{argv.minibatch_size}'
        f'_cg{argv.clip_grad}'
        f'_hd{"_".join(map(str, argv.hidden_dims))}'
        f'_sprs{"_".join(map(str, argv.sparsities))}'
        f'_d{argv.dropout}'
    )
    
    os.makedirs(argv.target_dir, exist_ok=True)
    with open(os.path.join(argv.target_dir, 'argv.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(vars(argv).items())
        
    if argv.clean_ckpt:
        print('No K-Fold')
        if os.path.exists(argv.target_dir):
            shutil.rmtree(argv.target_dir)
            print('Cleaned CKPT')
        else:
            print('First time training')
    
    os.makedirs(argv.target_dir, exist_ok=True)
    
    print(f'Model to be saved to path: {argv.target_dir}')
        
    return argv

if __name__=='__main__':
    # parse options and make directories
    argv = get_argv()


    if argv.no_kfold:
        from experiment_splits import train, test
        if not argv.no_train:
            print('Training..')
            train(argv)
        if not argv.no_test:
            print('Testing..')
            test(argv)
    else:
        from experiment import train, test
        # run and analyze experiment
        if not argv.no_train: train(argv)
        if not argv.no_test: test(argv)
        # if not argv.no_analysis: analyze(argv)
    exit(0)
