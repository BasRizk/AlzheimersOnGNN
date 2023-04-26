import util
import shutil

from experiment import train, test
# from analysis import analyze
import os

if __name__=='__main__':
    # parse options and make directories
    argv = util.option.parse()
    
    argv.targetdir = os.path.join(
        argv.targetdir,
        f'ws({argv.window_size})'
        f'_st({argv.window_stride})'
        f'_lr({argv.lr})mx({argv.max_lr})'
        f'_reg({argv.reg_lambda})',
        
        f'b({argv.minibatch_size})'
        f'_cg({argv.clip_grad})'
        f'_hd({"_".join(map(str, argv.hidden_dims))})'
        f'_sprs({"_".join(map(str, argv.sparsities))})'
        f'_d({argv.dropout})'
    )
    

    if argv.clean_ckpt:
        if os.path.exists(argv.targetdir):
            shutil.rmtree(argv.targetdir)
            print('Cleaned CKPT')
        else:
            print('First time training')
    
    os.makedirs(argv.targetdir, exist_ok=True)
    
    print(f'Model to be saved to path: {argv.targetdir}')

    if argv.clean_cache:
        print('Cleaning Cache..')
        for file in list(os.listdir(argv.data_dir)):
            if file.endswith('.pth'):
                os.remove(
                    os.path.join(argv.data_dir, file)
                )
                print(f'Cleaned Cache: {file}')


    
    if argv.no_kfold:
        from experiment_new import train
        train(argv)
    else:
        # run and analyze experiment
        if not argv.no_train: train(argv)
        if not argv.no_test: test(argv)
        # if not argv.no_analysis: analyze(argv)
    exit(0)
