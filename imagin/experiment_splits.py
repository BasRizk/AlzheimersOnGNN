import os
import util
import random
import torch
import numpy as np
from model import *
from dataset import ADNI
from tqdm import tqdm
from einops import repeat
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from experiment import step, process_input_data, summarize_results, log_fold, inference
from loguru import logger


def build_model(
    argv, 
    nums_nodes=None,
    device=None,
    steps_per_epoch=None,
    train=True
):    
    assert nums_nodes is not None
    assert device is not None
    assert steps_per_epoch is not None
    
    if train:
        os.makedirs(os.path.join(argv.targetdir, 'model', str(argv.k_fold)), exist_ok=True)

    # resume checkpoint if file exists
    if not train or os.path.isfile(os.path.join(argv.targetdir, 'checkpoint.pth')):
        logger.warning('resuming checkpoint experiment')
        checkpoint = torch.load(
            os.path.join(argv.targetdir, 'checkpoint.pth'),
            map_location=device
        )
    else:
        logger.info('Init ckpt for a new model')
        checkpoint = {
            'fold': 0,
            'epoch': 0,
            'model': None,
            'optimizer': None,
            'scheduler': None
        }
    
    # define model
    model = ModelIMAGIN(
        input_dims=nums_nodes,
        hidden_dims=argv.hidden_dims,
        num_classes=argv.num_classes,
        num_layers=argv.num_layers,
        sparsities=argv.sparsities,
    )
    if checkpoint['model'] is not None: 
        model.load_state_dict(checkpoint['model'])
        logger.debug(f'Loaded ckpt of model: {model}')
    elif not train:
        raise ValueError('No model checkpoint found')
    
    if not train:
        model.eval()
    model.to(device)
    

    if argv.num_classes > 1:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=argv.label_smoothing) 
        logger.debug('Set criterion to CrossEntropyLoss as num_classes > 1')
    else:
        criterion = torch.nn.MSELoss()
        logger.debug('Set criterion to MSELoss as num_classes <= 1')

    if train:
        # define optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=argv.lr)    
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=argv.max_lr, epochs=argv.num_epochs, 
            steps_per_epoch=steps_per_epoch,
            pct_start=0.2, div_factor=argv.max_lr/argv.lr, final_div_factor=1000
        )
        
        if checkpoint['optimizer'] is not None: 
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.debug(f'Loaded ckpt of optimizer: {optimizer}')
        if checkpoint['scheduler'] is not None:
            sch_dict = checkpoint['scheduler']
            sch_dict['total_steps'] =\
                sch_dict['total_steps'] +\
                argv.num_epochs * steps_per_epoch
            scheduler.load_state_dict(sch_dict)
            # scheduler.load_state_dict(checkpoint['scheduler'])
            logger.debug(f'Loaded ckpt of scheduler: {scheduler}, and set total_steps = {sch_dict["total_steps"]}')
    else:
        optimizer, scheduler = None, None

    return model, criterion, optimizer, scheduler, checkpoint


def train(argv):
    # make directories
    os.makedirs(os.path.join(argv.targetdir, 'model'), exist_ok=True)
    os.makedirs(os.path.join(argv.targetdir, 'summary'), exist_ok=True)

    # set seed and device
    torch.manual_seed(argv.seed)
    np.random.seed(argv.seed)
    random.seed(argv.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(argv.seed)
    else:
        device = torch.device("cpu")

    # define dataset
    dataset_train = ADNI(
        argv.data_dir,
        argv.data_type,
        argv.atlas_dir,
        argv.splits_dir,
        mode="Train",
        # dynamic_length=argv.dynamic_length, 
        k_fold=argv.k_fold,
        # smoothing_fwhm=argv.fwhm,
        num_classes=argv.num_classes,
        depth_of_slice=argv.depth_of_slice,
        slicing_stride=argv.slicing_stride,
        parallize_brains=argv.parallize_brains
    )

    dataset_val = ADNI(
        argv.data_dir,
        argv.data_type,
        argv.atlas_dir,
        argv.splits_dir,
        mode="Val",
        # dynamic_length=argv.dynamic_length, 
        k_fold=argv.k_fold,
        # smoothing_fwhm=argv.fwhm,
        num_classes=argv.num_classes,
        depth_of_slice=argv.depth_of_slice,
        slicing_stride=argv.slicing_stride,
        parallize_brains=argv.parallize_brains
    )

    n_workers=0  #4
    
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=argv.minibatch_size, shuffle=False,
        num_workers=n_workers,
        # pin_memory=True
    )
    
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=argv.minibatch_size, shuffle=False, 
        num_workers=n_workers,
        # pin_memory=True
    )


    # Start Experiment
    model, criterion, optimizer, scheduler, checkpoint = build_model(
        argv,
        nums_nodes=dataset_train.nums_nodes,
        device=device,
        steps_per_epoch=len(dataloader_train),
        train=True
    )
    
    # define logging objects
    summary_writer = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(argv.k_fold), 'train'))
    summary_writer_val = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(argv.k_fold), 'val'))
    logger_train = util.logger.LoggerIMAGIN(argv.k_fold, dataset_train.num_classes)
    logger_val = util.logger.LoggerIMAGIN(argv.k_fold, dataset_val.num_classes)


    best_accuracy = 0
    best_model = None
    # start training
    for epoch in range(checkpoint['epoch'], argv.num_epochs):
        logger_train.initialize(0)
        loss_accumulate = 0.0
        reg_ortho_accumulate = 0.0
        for i, x in enumerate(tqdm(dataloader_train, ncols=60, desc=f'k:{0} e:{epoch}')):
            # process input data
            all_dyn_a, all_sampling_endpoints, all_dyn_v, all_t, label =\
                process_input_data(
                    x,
                    device,
                    window_size=argv.window_size,
                    window_stride=argv.window_stride, 
                    nums_nodes=dataset_train.nums_nodes,
                    minibatch_size=argv.minibatch_size,
                    dynamic_length=argv.dynamic_length
                )

            logit, loss, attentions, latents, reg_ortho = step(
                model=model,
                criterion=criterion,
                all_dyn_v=all_dyn_v,
                all_dyn_a=all_dyn_a,
                all_sampling_endpoints=all_sampling_endpoints,
                all_t=all_t,
                label=label,
                reg_lambda=argv.reg_lambda,
                clip_grad=argv.clip_grad,
                device=device,
                optimizer=optimizer,
                scheduler=scheduler
            )
            
            pred = logit.argmax(1)
            prob = logit.softmax(1)
            
            if np.isnan(prob.detach().cpu().numpy()).any():
                logger.error('prob includes nan!!')
                breakpoint()
                
            loss_accumulate +=\
                loss.detach().cpu().numpy()
            reg_ortho_accumulate +=\
                reg_ortho.detach().cpu().numpy()
            logger_train.add(
                k=0, 
                pred=pred.detach().cpu().numpy(),
                true=label.detach().cpu().numpy(),
                prob=prob.detach().cpu().numpy()
            )
            summary_writer.add_scalar(
                'lr', scheduler.get_last_lr()[0], i+epoch*len(dataloader_train)
            )
        
        
        # summarize results
        samples, metrics = summarize_results(
            logger_train, summary_writer, 0,
            loss_accumulate, dataloader_train, reg_ortho_accumulate,
            epoch, attentions
        )
        logger.info(f'TRAIN: {metrics}, loss = {loss_accumulate}')

        # save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()},
            os.path.join(argv.targetdir, 'checkpoint.pth'))

        # check VALIDATION results
        metrics = inference( # VAL
            dataset=dataset_val,    
            dataloader=dataloader_val,
            k=0,
            model=model,
            criterion=criterion,
            device=device,
            argv=argv,
            logger=logger_val,
            summary_writer=summary_writer_val,
            str_pre_metrics="VAL",
            test_logging=False,
            set_fold=False
        )

        if metrics['accuracy'] > best_accuracy or (metrics['accuracy'] == best_accuracy and metrics['roc_auc'] > best_auroc):
            best_accuracy = metrics['accuracy']
            best_auroc = metrics['roc_auc']
            best_model = model.state_dict()
            torch.save(best_model, os.path.join(argv.targetdir, 'model', str(argv.k_fold), 'model.pth'))
            logger.success(f"BEST MODEL @ epoch {epoch}")


    checkpoint.update({'epoch': 0, 'model': None, 'optimizer': None, 'scheduler': None})

    # final validation results
    for atlas_name in dataset_val.atlases_names:
        os.makedirs(os.path.join(argv.targetdir, 'attention', atlas_name, str(argv.k_fold)), exist_ok=True)

    summary_writer.close()
    summary_writer_val.close()
    # summary_writer_test.close()
    # os.remove(os.path.join(argv.targetdir, 'checkpoint.pth'))


def test(argv):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    dataset_test = ADNI(
        argv.data_dir,
        argv.data_type,
        argv.atlas_dir,
        argv.splits_dir,
        mode="Test",
        # dynamic_length=argv.dynamic_length, 
        k_fold=0,
        # smoothing_fwhm=argv.fwhm,
        num_classes=argv.num_classes,
        depth_of_slice=argv.depth_of_slice,
        slicing_stride=argv.slicing_stride,
        parallize_brains=argv.parallize_brains
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, 
    )
    
    model, criterion, _, _, _ = build_model(
        argv,
        nums_nodes=dataset_test.nums_nodes,
        steps_per_epoch=len(dataloader_test),
        train=False
    )
    
    
       
    logger_test = util.logger.LoggerIMAGIN(argv.k_fold, dataset_test.num_classes)
    summary_writer_test = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(argv.k_fold), 'test'))
    inference( # TEST
        dataset=dataset_test,
        dataloader=dataloader_test,
        k=1,
        model=model,
        criterion=criterion,
        device=device,
        argv=argv,
        logger=logger_test,
        summary_writer=summary_writer_test,
        str_pre_metrics="Test set",
        test_logging=True,
        set_fold=False
    )

    # finalize experiment
    logger_test.to_csv(argv.targetdir)
    final_metrics = logger_test.evaluate()
    logger.success(f'final metric: {final_metrics}')

    torch.save(logger_test.get(), os.path.join(argv.targetdir, 'samples.pkl'))