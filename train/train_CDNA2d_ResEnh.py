# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# --------------------------------------------------------

import os
import time
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
import logging
import json
import random
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, ConstantLR

from dataset.hic_dna_dataset import prepare_datasets
from dataset.normalizer import Normalizer
from model.create_CDNA2d import create_model, create_loss

from config import *
import utils
from utils import *
from train.train_utils import *

torch.autograd.set_detect_anomaly(True)

def train(args):
    accelerator = Accelerator(
        gradient_accumulation_steps= args.accumulate_step,
        kwargs_handlers =  [DistributedDataParallelKwargs(find_unused_parameters = True)]
    )

    utils.VERBOSE = accelerator.is_main_process

    random.seed(args.seed + accelerator.process_index)
    np.random.seed(args.seed + accelerator.process_index)
    torch.manual_seed(args.seed + accelerator.process_index)
    torch.cuda.manual_seed(args.seed + accelerator.process_index)

    datestr = time.strftime('%m_%d_%H_%M')

    # out_dir: directory storing checkpoint files
    save_name = f'{datestr}_{args.method_name}_{args.resolution}'
    out_dir = os.path.join(save_dir, save_name)

    if accelerator.is_main_process:
        os.makedirs(out_dir, exist_ok=True)

        log_file_path = os.path.join(out_dir, "experiment.log")

        with open(os.path.join(out_dir, "args.json"), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(message)s',
            datefmt='%H:%M:%S',
            filename=log_file_path, 
            filemode='w', 
            force=True
        )

        vis_path = os.path.join(out_dir, 'vis')
        mkdir(vis_path)

        if not args.debug:
            if args.resume_wandb_run is not None:
                wandb.init(
                    project = "Cdiffusion",
                    id = args.resume_wandb_run,
                    resume = 'must',

                    config = args.__dict__
                )
            else:
                wandb.init(
                    project = "Cdiffusion",

                    config = args.__dict__
                )

    normalizer = Normalizer(args.normalization, max_reads=args.max_reads, denominator = args.denominator, step=args.step)

    model = create_model(**{**args.__dict__, 'normalizer' : normalizer, 'diffusion_steps' : 0})
    loss_fn = create_loss(**{**args.__dict__, 'normalizer' : normalizer, 'diffusion_steps' : 0})

    step = 0

    optimizer = AdamW(
                [
                    {'params': model.parameters(), 'lr' : args.base_learning_rate},
                ],
                betas = (0.9, 0.95)
            )

    max_step = args.max_step
    warmup_step = args.warmup_step
    mm = max_step*accelerator.num_processes
    wm = warmup_step*accelerator.num_processes

    if warmup_step>0:
        warmup_scheduler = LinearLR(optimizer, start_factor=1/wm, end_factor=1.0, total_iters=wm)
        scheduler = CosineAnnealingLR(optimizer, T_max=mm - wm) if args.lr_decay else ConstantLR(optimizer, factor=1, total_iters=10000000)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, scheduler], milestones=[wm])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=mm) if args.lr_decay else ConstantLR(optimizer, factor=1, total_iters=10000000)

    if accelerator.is_main_process:
        best = {}

    if args.load_checkpoint is not None:
        checkpoint = args.load_checkpoint
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)

        model.load_state_dict(state['model'])
        step = state['step']
        optimizer.load_state_dict(state['optimizer'])
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.base_learning_rate

        scheduler.load_state_dict(state['scheduler'])

        if accelerator.is_main_process:
            with open(os.path.join(os.path.dirname(checkpoint), 'best.json')) as f:
                best = json.load(f)

        print_info('loaded')
    elif args.initialize is not None:
        checkpoint = args.initialize
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        state_unified = {k.replace('unet', 'decoder'):v for k,v in state['model'].items()}
        model.load_state_dict(state_unified)

        print_info('model initialized')
    else:
        if args.initialize_dna_encoder is not None:
            state1 = torch.load(args.initialize_dna_encoder, map_location='cpu', weights_only=True)
            dna_encoder_state_dict = {
                k.replace('DNA_encoder.', ''): v
                for k, v in state1['model'].items()
                if k.startswith('DNA_encoder.')
            }

            model.DNA_encoder.load_state_dict(dna_encoder_state_dict)

            print('DNA_encoder initialized')

    accelerator.wait_for_everyone()

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    train_set, valid_set, sparse_valid_set = prepare_datasets(**{**args.__dict__, 'normalizer' : normalizer, 'dataset_size' : accelerator.num_processes * args.epoch_step * args.batch_size})

    train_dl = DataLoader(train_set, batch_size = args.batch_size, shuffle=True,  num_workers = 20, drop_last=True)
    valid_dl = DataLoader(valid_set, batch_size = args.batch_size, shuffle=False, num_workers = 20)

    train_dl, valid_dl = accelerator.prepare(train_dl, valid_dl)

    if sparse_valid_set is not None:
        sparse_valid_dl = DataLoader(sparse_valid_set, batch_size = args.batch_size, shuffle=False, num_workers = 8)
        sparse_valid_dl = accelerator.prepare(sparse_valid_dl)

    if accelerator.is_main_process:
        ckpt_maintainer = checkpoint_maintainer(debug = args.debug)

        print_info('Start training.')

    train_losses = []

    epoch = int(np.ceil((max_step-step) / len(train_dl)))
    for e in range(epoch):
        torch.cuda.empty_cache()
        model.train()

        pbar = tqdm(train_dl, disable = not accelerator.is_main_process)
        if args.eval_only:
            pbar=[]
        for data in pbar:
            with accelerator.accumulate(model):
                pred = model(**data)
                target = data['target_matrix']

                loss = loss_fn(pred.flatten(0,2), target.flatten(0,2))
                accelerator.backward(loss)

                if accelerator.is_main_process and step % 500 == 0:
                    vmin, vmax = normalizer.normalize(0), normalizer.normalize(100)
                    save_hic_batches_to_pdf(
                        pred,
                        target, 
                        data['positions'], 
                        save_path=os.path.join(vis_path, f'vis_{step}.pdf'), 
                        vmin=vmin,
                        vmax=vmax
                    )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step += 1

            train_losses.append(loss.cpu().item())

            learning_rate = scheduler.get_last_lr()[0]

            if accelerator.is_main_process and not args.debug:
                results = {
                    'train loss' : loss.cpu().item(),
                    'learning rate' : learning_rate
                }
                wandb.log(
                    results,
                    step = step
                )

            if accelerator.is_main_process:
                pbar.set_description(f'Epoch: {e}/{epoch}, Step: {step}, Avg loss: { np.mean(train_losses):.4e}, current loss: {np.mean(train_losses[-args.accumulate_step:]):.4e}')

        #start eval
        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

        model.eval()

        if accelerator.is_main_process:
            ckpt = os.path.join(out_dir, f'{step}.pt')
            state = {
                'model' : accelerator.unwrap_model(model).state_dict(),
                'step' : step,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }

            torch.save(state, ckpt)

            ckpt_maintainer.create_checkpoint(ckpt, keep = (step % 10000 == 0))

            ckpt_maintainer.link_checkpoint(ckpt, os.path.join(out_dir, f'last.pt'))

            learning_rate = scheduler.get_last_lr()[0]

            print_info(("Step: %d, \t Train, \t LR: %0.8f, \t Loss: %0.8f" % (step, learning_rate, np.mean(train_losses))))

            train_losses = []

        results = {}
        def val(dl, key_suffix = ''):
            with torch.no_grad():
                losses = []
                lmses =  [[] for _ in args.relative_resolutions]

                linear_normalizer = Normalizer('linear', 1000)

                valid_bar = tqdm(dl, disable = not accelerator.is_main_process)
                first = True
                for data in valid_bar:
                    pred = model(**data)
                    target = data['target_matrix']

                    losses.append(loss_fn(pred.flatten(0,2), target.flatten(0,2)).cpu().item())

                    sr = normalizer.unnormalize(pred.flatten(0,2), tensor=True).clamp(min=0)
                    hr = normalizer.unnormalize(target.flatten(0,2), tensor=True)

                    for i, pos in enumerate(data['positions'].flatten(0,2)):
                        if pos[0] == pos[4] and (pos[1] - pos[5])//args.resolution > -args.chunk:
                            sr = torch.triu(sr, (pos[1] - pos[5])//args.resolution)
                            hr = torch.triu(hr, (pos[1] - pos[5])//args.resolution)

                    if accelerator.is_main_process and first:
                        vmin, vmax = normalizer.normalize(0), normalizer.normalize(100)
                        save_hic_batches_to_pdf(
                            pred,
                            target, 
                            data['positions'], 
                            save_path=os.path.join(vis_path, f'vis_val_{step}.pdf'), 
                            vmin=vmin, 
                            vmax=vmax
                        )
                    if first:
                        accelerator.wait_for_everyone()

                    first = False

                    for i, r in enumerate(args.relative_resolutions):
                        sr_r = torch.nn.functional.avg_pool2d(sr, r ,r)
                        hr_r = torch.nn.functional.avg_pool2d(hr, r ,r)

                        sr_r = torch.triu(sr_r, diagonal = 1)
                        hr_r = torch.triu(hr_r, diagonal = 1)

                        sr_rln = linear_normalizer.normalize(sr_r * r * r, tensor = True)
                        hr_rln = linear_normalizer.normalize(hr_r * r * r, tensor = True)

                        sr_rln = sr_rln.clamp(min=0, max=1)
                        hr_rln = hr_rln.clamp(min=0, max=1)

                        lmse = ((sr_rln - hr_rln)**2).mean()

                        lmses[i].append(lmse)

            losses  = torch.tensor(losses).to(device=accelerator.device)
            lmses = torch.tensor(lmses).to(device=accelerator.device).transpose(0,1)

            gathered_losses : torch.Tensor = accelerator.gather(losses)
            gathered_lmses : torch.Tensor = accelerator.gather(lmses)
            
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                gathered_losses = gathered_losses.mean(dim=0).cpu().numpy()
                kloss = f'valid loss{key_suffix}'
                loss = float(gathered_losses)
                results[kloss] = loss
                print_info(f"Step: {step}, \t valid {kloss} is {loss}")
                if loss < best.get(kloss, np.inf):
                    best[kloss] = loss
                    ckpt_maintainer.link_checkpoint(ckpt, os.path.join(out_dir, f'best_loss{key_suffix}.pt'))

                gathered_lmses = gathered_lmses.mean(dim=0).cpu().numpy()
                for i, r in enumerate(args.relative_resolutions):
                    res = r*args.resolution
                    klmse = f'{res}[LMSE]{key_suffix}'
                    lmse = float(gathered_lmses[i])
                    results[klmse] = lmse
                    print_info(f"Step: {step}, \t valid {klmse} is {lmse}")
                    if lmse < best.get(klmse, np.inf):
                        best[klmse] = lmse
                        ckpt_maintainer.link_checkpoint(ckpt, os.path.join(out_dir, f'best_lmse{key_suffix}_{res}.pt'))

        val(valid_dl)
        if sparse_valid_set is not None:
            val(sparse_valid_dl, key_suffix = '_Sparse')

        if accelerator.is_main_process:
            with open(os.path.join(out_dir, "best.json"), 'w') as f:
                json.dump(best, f, indent=2)
            ckpt_maintainer.clean()

            if not args.debug:
                wandb.log(
                    results,
                    step=step
                )

        if args.eval_only:
            print_info('Evaluation finished.')
            break
        
    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == '__main__':
    parser = train_parser()
    parser.add_argument('--initialize-dna-encoder', type=str, default=None)

    parser.set_defaults(
        method_name = 'CDNA2d-ResEnh',
        sample=True,
        max_step=200000
    )
    args = parser.parse_args()

    train(args)