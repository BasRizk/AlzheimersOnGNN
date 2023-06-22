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


def step(model, criterion, all_dyn_v, all_dyn_a, all_sampling_endpoints, all_t, label, reg_lambda, clip_grad=0.0, device='cpu', optimizer=None, scheduler=None):
    if optimizer is None: model.eval()
    else: model.train()

    # run model
    logit, attention, latent, reg_ortho = model(all_dyn_v, all_dyn_a, all_t, all_sampling_endpoints)

    loss = criterion(logit, label)
    reg_ortho *= reg_lambda
    loss += reg_ortho

    # optimize model
    if optimizer is not None:
       optimizer.zero_grad()
       loss.backward()
       if clip_grad > 0.0: torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad)
       optimizer.step()
       if scheduler is not None:
           scheduler.step()
    # else:
    #     breakpoint()

    return logit, loss, attention, latent, reg_ortho


def process_input_data(
        x,
        device,
        window_size=None,
        window_stride=None,
        minibatch_size=None,
        nums_nodes=None,
        dynamic_length=None
    ):
    all_dyn_a, all_sampling_endpoints, all_dyn_v, all_t = [], [], [], []
    for r_i, x_ss in enumerate(x['slices_per_atlas']):
        dyn_a, sampling_points = util.bold.process_dynamic_fc(
            x_ss, 
            window_size,
            window_stride, 
            dynamic_length=dynamic_length
        )
        
        sampling_endpoints = [
            p + window_size for p in sampling_points
        ]
        
        # ARCHIVED OLD LOGIC
        # TRAINING: if i==0:
        # VALIDATING: if not dyn_v.shape[1] == dyn_a.shape[1]: 
        # FINAL VALIDATING/TESTING: BOTH (I think a mistake)
        
        dyn_v = repeat(
            torch.eye(nums_nodes[r_i]),
            'n1 n2 -> b t n1 n2', 
            t=len(sampling_points),
            b=minibatch_size
        )
            
        if len(dyn_a) < minibatch_size:
            dyn_v = dyn_v[:len(dyn_a)]
            
        t = x_ss.permute(1,0,2)
        
        all_dyn_a.append(dyn_a.to(device))
        all_sampling_endpoints.append(sampling_endpoints)
        all_dyn_v.append(dyn_v.to(device))
        all_t.append(t.to(device))
    
    # breakpoint()
    return all_dyn_a, all_sampling_endpoints, all_dyn_v, all_t, x['label'].to(device)

def summarize_results(
                logger, summary_writer, k,
                loss_accumulate, dataloader, reg_ortho_accumulate,
                epoch, attentions
            ):
    samples = logger.get(k)
    metrics = logger.evaluate(k)
    summary_writer.add_scalar('loss', loss_accumulate/len(dataloader), epoch)
    summary_writer.add_scalar('reg_ortho', reg_ortho_accumulate/len(dataloader), epoch)
    summary_writer.add_pr_curve('precision-recall', samples['true'], samples['prob'][:,1], epoch)
    [summary_writer.add_scalar(key, value, epoch) for key, value in metrics.items() if not key=='fold']

    # TODO think what to do instead about this
    for atlas_name, att in attentions.items():
        for key, value in att.items():
            summary_writer.add_image(
                f"{atlas_name}_{key}",
                make_grid(
                    value[-1].unsqueeze(1), 
                    normalize=True, scale_each=True
                ),
                epoch
            )
    summary_writer.flush()
    return samples, metrics

def log_fold(logger, target_dir, k, fold_attentions, latent_accumulates):
    os.makedirs(target_dir, exist_ok=True)
    logger.to_csv(target_dir, k)
    for atlas_name, att in fold_attentions.items():
        save_dir = os.path.join(target_dir, 'attention', atlas_name, str(k))
        os.makedirs(save_dir, exist_ok=True)
        for key, value in att.items():
            np.save(
                os.path.join(save_dir, f'{key}.npy'), 
                np.concatenate(value)
            )
    
    for atlas_name, lat in latent_accumulates.items():
        np.save(
            os.path.join(target_dir, 'attention', atlas_name, str(k), 'latent.npy'),
            np.concatenate(lat)
        )



def inference(
        dataset=None,
        dataloader=None,
        k=None,
        model=None,
        criterion=None,
        device=None,
        argv=None,
        logger=None,
        summary_writer=None,
        str_pre_metrics="",
        test_logging=True, # FOR FILE LOGGING
        set_fold=True
    ):
    
    if test_logging:
        # define logging objects
        fold_attentions = {
            atlas_name: {'node_attention': [], 'time_attention': []}
            for atlas_name in dataset.atlases_names
        }
        latent_accumulates = {
            atlas_name: []
            for atlas_name in dataset.atlases_names
        }

    logger.initialize(k)
    if set_fold:
        dataset.set_fold(k, train=False)
    loss_accumulate = 0.0
    reg_ortho_accumulate = 0.0
    
    for i, x in enumerate(tqdm(dataloader, ncols=60, desc=f'k:{k}')):
        with torch.no_grad():
            # process input data
            all_dyn_a, all_sampling_endpoints, all_dyn_v, all_t, label =\
                process_input_data(
                    x, 
                    device,
                    window_size=argv.window_size,
                    window_stride=argv.window_stride, 
                    minibatch_size=argv.minibatch_size,
                    nums_nodes=dataset.nums_nodes,
                    dynamic_length=None
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
                optimizer=None,
                scheduler=None
            )
            pred = logit.argmax(1)
            prob = logit.softmax(1)
            logger.add(
                k=k, pred=pred.detach().cpu().numpy(), 
                true=label.detach().cpu().numpy(), 
                prob=prob.detach().cpu().numpy()
            )

            loss_accumulate += loss.detach().cpu().numpy()
            reg_ortho_accumulate += reg_ortho.detach().cpu().numpy()

            if test_logging:
                for (atlas_name, att), (_, lat) in zip(attentions.items(), latents.items()):
                    fold_attentions[atlas_name]['node_attention'].append(
                        att['node-attention'].detach().cpu().numpy()
                    )
                    fold_attentions[atlas_name]['time_attention'].append(
                        att['time-attention'].detach().cpu().numpy()
                    )
                    latent_accumulates[atlas_name].append(lat.detach().cpu().numpy())

    # summarize results
    samples, metrics = summarize_results(
        logger, summary_writer, k,
        loss_accumulate, dataloader, reg_ortho_accumulate,
        None, attentions
    )
    print(str_pre_metrics, metrics, f'loss = {loss_accumulate}')

    if test_logging:
        # finalize fold
        log_fold(
            logger=logger,
            target_dir=argv.targetdir, 
            k=k, 
            fold_attentions=fold_attentions, 
            latent_accumulates=latent_accumulates
        )
        # breakpoint()
        # del fold_attentions

    return metrics
            


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
    dataset = ADNI(
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
        device=device,
        parallize_brains=argv.parallize_brains
    )

    n_workers=0 #4
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=argv.minibatch_size, shuffle=False,
        num_workers=n_workers,
        # pin_memory=True
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, 
        num_workers=n_workers,
        # pin_memory=True
    )

    logger_test = util.logger.LoggerIMAGIN(argv.k_fold, dataset.num_classes)

    # resume checkpoint if file exists
    if os.path.isfile(os.path.join(argv.targetdir, 'checkpoint.pth')):
        print('resuming checkpoint experiment')
        checkpoint = torch.load(os.path.join(argv.targetdir, 'checkpoint.pth'), map_location=device)
    else:
        checkpoint = {
            'fold': 0,
            'epoch': 0,
            'model': None,
            'optimizer': None,
            'scheduler': None
        }
    
    # start experiment
    for k in range(checkpoint['fold'], argv.k_fold):
        # make directories per fold
        os.makedirs(os.path.join(argv.targetdir, 'model', str(k)), exist_ok=True)

        # set dataloader
        dataset.set_fold(k, train=True)

        # define model
        model = ModelIMAGIN(
            input_dims=dataset.nums_nodes,
            hidden_dims=argv.hidden_dims,
            num_classes=dataset.num_classes,
            num_layers=argv.num_layers,
            sparsities=argv.sparsities,
        )

        model.to(device)
        if checkpoint['model'] is not None: model.load_state_dict(checkpoint['model'])
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=argv.label_smoothing) if dataset_test.num_classes > 1 else torch.nn.MSELoss()


        # define optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=argv.lr)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=argv.max_lr, epochs=argv.num_epochs, 
            steps_per_epoch=len(dataloader),
            pct_start=0.2, div_factor=argv.max_lr/argv.lr, final_div_factor=1000
        )
        if checkpoint['optimizer'] is not None: optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['scheduler'] is not None: scheduler.load_state_dict(checkpoint['scheduler'])

        # define logging objects
        summary_writer = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(k), 'train'), )
        summary_writer_val = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(k), 'val'), )
        logger = util.logger.LoggerIMAGIN(argv.k_fold, dataset.num_classes)
        logger_val = util.logger.LoggerIMAGIN(argv.k_fold, dataset.num_classes)

        best_accuracy = 0
        best_model = None
        # start training
        for epoch in range(checkpoint['epoch'], argv.num_epochs):
            logger.initialize(k)
            dataset.set_fold(k, train=True)
            loss_accumulate = 0.0
            reg_ortho_accumulate = 0.0
            for i, x in enumerate(tqdm(dataloader, ncols=60, desc=f'k:{k} e:{epoch}')):
                # process input data
                all_dyn_a, all_sampling_endpoints, all_dyn_v, all_t, label =\
                    process_input_data(
                        x,
                        device,
                        window_size=argv.window_size,
                        window_stride=argv.window_stride, 
                        nums_nodes=dataset.nums_nodes,
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
                loss_accumulate += loss.detach().cpu().numpy()
                reg_ortho_accumulate += reg_ortho.detach().cpu().numpy()
                logger.add(k=k, pred=pred.detach().cpu().numpy(), true=label.detach().cpu().numpy(), prob=prob.detach().cpu().numpy())
                summary_writer.add_scalar('lr', scheduler.get_last_lr()[0], i+epoch*len(dataloader))

            # summarize results
            samples, metrics = summarize_results(
                logger, summary_writer, k,
                loss_accumulate, dataloader, reg_ortho_accumulate,
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
                os.path.join(argv.targetdir, 'checkpoint.pth')
            )

            # check validation results
            metrics = inference( # VAL
                dataset=dataset,    
                dataloader=dataloader_test,
                k=k,
                model=model,
                criterion=criterion,
                device=device,
                argv=argv,
                logger=logger_val,
                summary_writer=summary_writer_val,
                str_pre_metrics="VAL",
                test_logging=False
            )

            if metrics['accuracy'] > best_accuracy or (metrics['accuracy'] == best_accuracy and metrics['roc_auc'] > best_auroc):
                best_accuracy = metrics['accuracy']
                best_auroc = metrics['roc_auc']
                best_model = model.state_dict()
                print("BEST MODEL")

        # finalize fold
        if best_model is not None:
            torch.save(best_model, os.path.join(argv.targetdir, 'model', str(k), 'model.pth'))
        else:
            print('MODEL IS NONE, TO BE LOADED')
            epoch = argv.num_epochs
            # k = argv.k_fold

        checkpoint.update({'epoch': 0, 'model': None, 'optimizer': None, 'scheduler': None})

        # final validation results
        for atlas_name in dataset.atlases_names:
            os.makedirs(os.path.join(argv.targetdir, 'attention', atlas_name, str(k)), exist_ok=True)

        model.load_state_dict(torch.load(os.path.join(argv.targetdir, 'model', str(k), 'model.pth')))
        

        summary_writer_test = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(k), 'test'))
        inference( # TEST
            dataset=dataset,
            dataloader=dataloader_test,
            k=k,
            model=model,
            criterion=criterion,
            device=device,
            argv=argv,
            logger=logger_test,
            summary_writer=summary_writer_test,
            str_pre_metrics="FINAL VAL",
            test_logging=True
        )

    # finalize experiment
    logger_test.to_csv(argv.targetdir)
    final_metrics = logger_test.evaluate()
    print(final_metrics, f'loss = {loss_accumulate}')

    torch.save(logger_test.get(), os.path.join(argv.targetdir, 'samples.pkl'))

    summary_writer.close()
    summary_writer_val.close()
    summary_writer_test.close()
    # os.remove(os.path.join(argv.targetdir, 'checkpoint.pth'))

def test(argv):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # define dataset
    dataset: ADNI = ADNI(
        argv.data_dir,
        argv.data_type,
        argv.atlas_dir,
        argv.splits_dir,
        # dynamic_length=argv.dynamic_length, 
        k_fold=argv.k_fold,
        # smoothing_fwhm=argv.fwhm,
        num_classes=argv.num_classes,
        depth_of_slice=argv.depth_of_slice,
        slicing_stride=argv.slicing_stride,
        parallize_brains=argv.parallize_brains
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=argv.minibatch_size, shuffle=False)
    logger = util.logger.LoggerIMAGIN(argv.k_fold, dataset.num_classes)

    for k in range(argv.k_fold):

        for atlas_name in dataset.atlases_names:
            os.makedirs(os.path.join(argv.targetdir, 'attention', atlas_name, str(k)), exist_ok=True)

        model = ModelIMAGIN(
            input_dims=dataset.nums_nodes,
            hidden_dims=argv.hidden_dims,
            num_classes=dataset.num_classes,
            num_layers=argv.num_layers,
            sparsities=argv.sparsities,
        )
        model.to(device)

        model.load_state_dict(torch.load(os.path.join(argv.targetdir, 'model', str(k), 'model.pth')))
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=argv.label_smoothing) if dataset_train.num_classes > 1 else torch.nn.MSELoss()
        summary_writer = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(k), 'test'))

        inference(
            dataset=dataset,
            dataloader=dataloader,
            k=k,
            model=model,
            criterion=criterion,
            device=device,
            argv=argv,
            logger=logger,
            summary_writer=summary_writer,
            test_logging=True
        )   
        
        
    # finalize experiment
    logger.to_csv(argv.targetdir)
    final_metrics = logger.evaluate()
    print(final_metrics)
    summary_writer.close()
    torch.save(logger.get(), os.path.join(argv.targetdir, 'samples.pkl'))
