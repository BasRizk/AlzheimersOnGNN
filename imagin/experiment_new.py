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
        k_fold=None,
        # smoothing_fwhm=argv.fwhm,
        num_classes=argv.num_classes,
        depth_of_slice=argv.depth_of_slice,
        slicing_stride=argv.slicing_stride,
        device=device,
        parallize_brains=argv.parallize_brains
    )

    dataset_val = ADNI(
        argv.data_dir,
        argv.data_type,
        argv.atlas_dir,
        argv.splits_dir,
        mode="Val",
        # dynamic_length=argv.dynamic_length, 
        k_fold=None,
        # smoothing_fwhm=argv.fwhm,
        num_classes=argv.num_classes,
        depth_of_slice=argv.depth_of_slice,
        slicing_stride=argv.slicing_stride,
        device=device,
        parallize_brains=argv.parallize_brains
    )

    n_workers=0  #4
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=argv.minibatch_size, shuffle=False,
        num_workers=n_workers,
        # pin_memory=True
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False, 
        num_workers=n_workers,
        # pin_memory=True
    )
    

    # Start Experiment
    k=0 # TODO kept only to reduce modifications

    # resume checkpoint if file exists
    if os.path.isfile(os.path.join(argv.targetdir, 'checkpoint.pth')):
        print('resuming checkpoint experiment')
        checkpoint = torch.load(
            os.path.join(argv.targetdir, 'checkpoint.pth'),
            map_location=device
        )
    else:
        checkpoint = {
            'fold': 0,
            'epoch': 0,
            'model': None,
            'optimizer': None,
            'scheduler': None
        }
    

    os.makedirs(os.path.join(argv.targetdir, 'model', str(k)), exist_ok=True)

    # define model
    model = ModelIMAGIN(
        input_dims=dataset_train.nums_nodes,
        hidden_dims=argv.hidden_dims,
        num_classes=dataset_train.num_classes,
        num_layers=argv.num_layers,
        sparsities=argv.sparsities,
    )

    model.to(device)
    if checkpoint['model'] is not None: model.load_state_dict(checkpoint['model'])
    criterion = torch.nn.CrossEntropyLoss() if dataset_train.num_classes > 1 else torch.nn.MSELoss()


    # define optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=argv.lr)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=argv.max_lr, epochs=argv.num_epochs, 
        steps_per_epoch=len(dataloader_train),
        pct_start=0.2, div_factor=argv.max_lr/argv.lr, final_div_factor=1000
    )
    if checkpoint['optimizer'] is not None: optimizer.load_state_dict(checkpoint['optimizer'])
    if checkpoint['scheduler'] is not None: scheduler.load_state_dict(checkpoint['scheduler'])

    # define logging objects
    summary_writer = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(k), 'train'), )
    summary_writer_val = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(k), 'val'), )
    logger_train = util.logger.LoggerIMAGIN(argv.k_fold, dataset_train.num_classes)
    logger_val = util.logger.LoggerIMAGIN(argv.k_fold, dataset_val.num_classes)

    best_accuracy = 0
    best_model = None
    # start training
    for epoch in range(checkpoint['epoch'], argv.num_epochs):
        logger_train.initialize(k)
        loss_accumulate = 0.0
        reg_ortho_accumulate = 0.0
        for i, x in enumerate(tqdm(dataloader_train, ncols=60, desc=f'k:{k} e:{epoch}')):
            # process input data
            # breakpoint()
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
                print('nan')
            loss_accumulate +=\
                loss.detach().cpu().numpy()
            reg_ortho_accumulate +=\
                reg_ortho.detach().cpu().numpy()
            logger_train.add(
                k=k, pred=pred.detach().cpu().numpy(), true=label.detach().cpu().numpy(), prob=prob.detach().cpu().numpy()
            )
            summary_writer.add_scalar(
                'lr', scheduler.get_last_lr()[0], i+epoch*len(dataloader_train)
            )
        # summarize results
        samples, metrics = summarize_results(
            logger_train, summary_writer, k,
            loss_accumulate, dataloader_train, reg_ortho_accumulate,
            epoch, attentions
        )
        print("TRAIN:", metrics, f'loss = {loss_accumulate}')

        # save checkpoint
        torch.save({
            'fold': k,
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()},
            os.path.join(argv.targetdir, 'checkpoint.pth'))

        # check VALIDATION results
        metrics = inference( # VAL
            dataset=dataset_val,    
            dataloader=dataloader_val,
            k=k,
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
            torch.save(best_model, os.path.join(argv.targetdir, 'model', str(k), 'model.pth'))
            print("BEST MODEL")


    checkpoint.update({'epoch': 0, 'model': None, 'optimizer': None, 'scheduler': None})

    # final validation results
    for atlas_name in dataset_val.atlases_names:
        os.makedirs(os.path.join(argv.targetdir, 'attention', atlas_name, str(k)), exist_ok=True)

    model.load_state_dict(torch.load(os.path.join(argv.targetdir, 'model', str(k), 'model.pth')))

    # TODO   
    # logger_test = util.logger.LoggerIMAGIN(argv.k_fold, dataset.num_classes)
    # summary_writer_test = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(k), 'test'))
    # inference( # TEST
    #     dataset=dataset,
    #     dataloader=dataloader_test,
    #     k=k,
    #     model=model,
    #     criterion=criterion,
    #     device=device,
    #     argv=argv,
    #     logger=logger_test,
    #     summary_writer=summary_writer_test,
    #     str_pre_metrics="FINAL VAL",
    #     test_logging=True,
    #     set_fold=False
    # )

    # finalize experiment
    # logger_test.to_csv(argv.targetdir)
    # final_metrics = logger_test.evaluate()
    # print(final_metrics)

    # torch.save(logger_test.get(), os.path.join(argv.targetdir, 'samples.pkl'))

    summary_writer.close()
    summary_writer_val.close()
    # summary_writer_test.close()
    # os.remove(os.path.join(argv.targetdir, 'checkpoint.pth'))
