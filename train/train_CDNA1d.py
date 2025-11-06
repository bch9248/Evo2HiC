# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# --------------------------------------------------------

import os
import time
import numpy as np
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import logging
import json
import random
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from dataset.hic_dna_track_dataset import *
from dataset.normalizer import Normalizer
from model.create_CDNA1d import create_model

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

    model = create_model(**{**args.__dict__, 'normalizer' : normalizer})

    step = 0

    optimizer = Adam(
                [
                    {'params': model.parameters(), 'lr' : args.base_learning_rate},
                ],
                betas=(0.9, 0.95)
            )

    max_step = args.max_step
    warmup_step = args.warmup_step
    mm = max_step*accelerator.num_processes
    wm = warmup_step*accelerator.num_processes

    if warmup_step>0:
        warmup_scheduler = LinearLR(optimizer, start_factor=1/wm, end_factor=1.0, total_iters=wm)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=mm - wm)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[wm])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=mm)

    if accelerator.is_main_process:
        best = {}

    if args.load_checkpoint is not None:
        checkpoint = args.load_checkpoint
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)

        model.load_state_dict(state['model'])
        step = state['step']
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])

        if accelerator.is_main_process:
            with open(os.path.join(os.path.dirname(checkpoint), 'best.json')) as f:
                best = json.load(f)

        print_info('loaded')
    elif args.initialize is not None:
        checkpoint = args.initialize
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)

        model_state = model.state_dict()
        new_state_dict = {}

        # Handle DNA_encoder.encoder → DNA_encoder
        for k, v in state['model'].items():
            if k.startswith("DNA_encoder.encoder."):
                new_key = "DNA_encoder." + k[len("DNA_encoder.encoder."):]
                if new_key in model_state and v.shape == model_state[new_key].shape:
                    new_state_dict[new_key] = v
                else:
                    print(f"[DNA] Skipping {k} → {new_key}: shape mismatch or key not in model.")

        # Handle HiC_encoder directly
        for k, v in state['model'].items():
            if k.startswith("HiC_encoder."):
                if k in model_state and v.shape == model_state[k].shape:
                    new_state_dict[k] = v
                else:
                    print(f"[HiC] Skipping {k}: shape mismatch or key not in model.")

        # Update and load
        model_state.update(new_state_dict)
        model.load_state_dict(model_state)

        print_info('initialized')

    accelerator.wait_for_everyone()

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    train_set, valid_set = prepare_datasets(**{**args.__dict__, 'normalizer' : normalizer, 'dataset_size' : accelerator.num_processes * args.epoch_step * args.batch_size})

    train_dl = DataLoader(train_set, batch_size = args.batch_size, shuffle=True,  num_workers = 20, drop_last=True)
    valid_dl = DataLoader(valid_set, batch_size = args.batch_size, shuffle=False, num_workers = 20)

    train_dl, valid_dl = accelerator.prepare(train_dl, valid_dl)

    if accelerator.is_main_process:
        ckpt_maintainer = checkpoint_maintainer(debug = args.debug)

        print_info('Start training.')

    train_losses = []


    epoch = int(np.ceil((max_step-step) / len(train_dl)))
    for e in range(epoch):
        torch.cuda.empty_cache()

        model.train()
        pbar = tqdm(train_dl, disable = not accelerator.is_main_process)
        for data in pbar:
            with accelerator.accumulate(model):
                pred = model(**data)
                target = data['track']
                loss0 = ((pred-target)**2).mean()
                pred_n = torch.nn.functional.normalize(pred, dim=-1)
                target_n = torch.nn.functional.normalize(target, dim=-1)
                loss1 = (1- (pred_n*target_n).sum(dim=-1)).mean()
                loss = loss0 + loss1
                accelerator.backward(loss)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step += 1

            train_losses.append(loss.cpu().item())

            learning_rate = scheduler.get_last_lr()[0]

            if accelerator.is_main_process and not args.debug:
                wandb.log(
                    {
                        'train loss (mse)' : loss0.cpu().item(),
                        'train loss (cos)' : loss1.cpu().item(),
                        'train loss (sum)' : loss.cpu().item(),
                        'learning rate' : learning_rate
                    },
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

            print_info(("Step: %d, \t Train, \t LR: %0.8f, \t Loss: %0.8f" % (step, learning_rate, np.mean(train_losses))))

            train_losses = []

        results = {}
        def val(dl, key_suffix = ''):
            with torch.no_grad():
                mses = [[] for _, _ in enumerate(tracks)]
                pccs = [[] for _, _ in enumerate(tracks)]

                valid_bar = tqdm(dl, disable = not accelerator.is_main_process)
                for data in valid_bar:
                    pred = model(**data).flatten(0,2)
                    target = data['track'].flatten(0,2)

                    for j in range(pred.shape[0]):
                        for i, _ in enumerate(tracks):
                            mse = ((pred[j, i] - target[j, i])**2).mean()
                            mses[i].append(mse)

                            pcc = np.nan_to_num(pearsonr(pred[j, i].cpu().numpy(), target[j, i].cpu().numpy())[0])
                            pccs[i].append(pcc)

                mses  = torch.tensor(mses).to(device=accelerator.device).transpose(0,1)
                pccs = torch.tensor(pccs).to(device=accelerator.device).transpose(0,1)

                gathered_mses : torch.Tensor = accelerator.gather(mses).mean(dim=0)
                gathered_pccs : torch.Tensor = accelerator.gather(pccs).mean(dim=0)

            if accelerator.is_main_process:
                gathered_mses = gathered_mses.cpu().numpy()
                for i, t in enumerate(tracks):
                    kmse = f'{t}[MSE]{key_suffix}'
                    mse = float(gathered_mses[i])
                    results[kmse] = mse
                    print_info(f"Step: {step}, \t valid {kmse} is {mse}")

                    if mse < best.get(kmse, np.inf):
                        best[kmse] = mse
                        ckpt_maintainer.link_checkpoint(ckpt, os.path.join(out_dir, f'best_mse{key_suffix}_{t}.pt'))

                gathered_pccs = gathered_pccs.cpu().numpy()
                for i, t in enumerate(tracks):
                    kpcc = f'{t}[PCC]{key_suffix}'
                    pcc = float(gathered_pccs[i])
                    results[kpcc] = pcc
                    print_info(f"Step: {step}, \t valid {kpcc} is {pcc}")

                    if pcc > best.get(kpcc, 0):
                        best[kpcc] = pcc
                        ckpt_maintainer.link_checkpoint(ckpt, os.path.join(out_dir, f'best_pcc{key_suffix}_{t}.pt'))

        val(valid_dl)

        if accelerator.is_main_process:
            with open(os.path.join(out_dir, "best.json"), 'w') as f:
                json.dump(best, f, indent=2)
            ckpt_maintainer.clean()

            if not args.debug:
                wandb.log(
                    results,
                    step=step
                )

    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == '__main__':
    parser = train_parser()
    parser.add_argument('--track-input-dim', type=int, default = 256, help='The number of dimensions at the first layer in track decoder')
    parser.set_defaults(
        method_name = 'CDNAtrack',

        sample = True,
        flip_prob = 0.5,
        DNA_shift = 100
    )
    args = parser.parse_args()

    train(args)