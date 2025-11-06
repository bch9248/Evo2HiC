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
from model.siglip import Siglip, siglip_HiC_DNA, siglip_DNA_evo2

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
    siglip = Siglip()

    if args.evo2_option == 'Yes':
        siglip2 = Siglip()
        projection0 = nn.Linear(args.emb_dim, args.evo2_projection_dim)
        projection1 = nn.Linear(evo2_hidden_size * 2, args.evo2_projection_dim)

    step = 0

    optimizer = AdamW(
                [
                    {'params': model.parameters(), 'lr' : args.base_learning_rate },
                    {'params': siglip.parameters(), 'lr' : args.base_learning_rate}
                ] + ([
                    {'params': siglip2.parameters(), 'lr' : args.base_learning_rate},
                    {'params': projection0.parameters(), 'lr' : args.base_learning_rate},
                    {'params': projection1.parameters(), 'lr' : args.base_learning_rate}
                ] if args.evo2_option == 'Yes' else []),
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
        siglip.load_state_dict(state['siglip'])
        step = state['step']
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        if 'siglip' in state:
            siglip.load_state_dict(state['siglip'])
        if args.evo2_option == 'Yes':
            if 'siglip2' in state:
                siglip2.load_state_dict(state['siglip2'])
            if 'projection0' in state:
                projection0.load_state_dict(state['projection0'])
            if 'projection1' in state:
                projection1.load_state_dict(state['projection1'])

        if accelerator.is_main_process:
            with open(os.path.join(os.path.dirname(checkpoint), 'best.json')) as f:
                best = json.load(f)

        print_info('loaded')
    elif args.initialize is not None:
        checkpoint = args.initialize
        state = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(state['model'])

        if 'siglip' in state:
            siglip.load_state_dict(state['siglip'])
        if args.evo2_option == 'Yes':
            if 'siglip2' in state:
                siglip2.load_state_dict(state['siglip2'])
            if 'projection0' in state:
                projection0.load_state_dict(state['projection0'])
            if 'projection1' in state:
                projection1.load_state_dict(state['projection1'])
        print_info('model initialized')

    accelerator.wait_for_everyone()

    if args.evo2_option == 'Yes':
        model, siglip, siglip2, projection0, projection1, optimizer, scheduler = accelerator.prepare(model, siglip, siglip2, projection0, projection1, optimizer, scheduler)
    else:
        model, siglip, optimizer, scheduler = accelerator.prepare(model, siglip, optimizer, scheduler)

    train_set, valid_set, _ = prepare_datasets(**{**args.__dict__, 'normalizer' : normalizer, 'dataset_size' : accelerator.num_processes * args.epoch_step * args.batch_size})

    train_dl = DataLoader(train_set, batch_size = args.batch_size, shuffle=True,  num_workers = 20, drop_last=True)
    valid_dl = DataLoader(valid_set, batch_size = args.batch_size, shuffle=False, num_workers = 20)

    train_dl, valid_dl = accelerator.prepare(train_dl, valid_dl)

    if accelerator.is_main_process:
        ckpt_maintainer = checkpoint_maintainer(debug = args.debug)

        print_info('Start training.')

    train_losses = []

    boundary = (args.chunk - args.stride)//2

    if args.evo2_option == 'Yes' and args.evo2_negative_sample > 0:
        min_candidates = args.evo2_negative_sample + args.batch_size * args.train_pos_per_row
        max_candidates = args.evo2_negative_sample * 20
        evo2_negative_poses, evo2_negative_embeds = [], []
        for data in train_dl:
            for i in range(data['positions'].shape[0]):
                for j in range(data['positions'].shape[1]):
                    evo2_negative_poses.append(data['positions'][i, j, ..., 4:7].cpu())
                    evo2_negative_embeds.append(cut1d(data['embedding_row'][i,j], boundary).cpu())
                    if len(evo2_negative_embeds) >= min_candidates:
                        break
            if len(evo2_negative_embeds) >= min_candidates:
                break

    epoch = int(np.ceil((max_step-step) / len(train_dl)))
    for e in range(epoch):
        torch.cuda.empty_cache()

        model.train()
        siglip.train()
        if args.evo2_option == 'Yes':
            siglip2.train()
            projection0.train()
            projection1.train()

        pbar = tqdm(train_dl, disable = not accelerator.is_main_process)
        for data in pbar:
            with accelerator.accumulate(model):
                HiC_embeds, DNA_embeds, DNA_row_embeds = model(**data, return_emb_directly = True)

                HiC_embeds = cut2d(HiC_embeds, boundary)
                DNA_embeds = cut2d(DNA_embeds, boundary)

                loss0 = siglip_HiC_DNA(HiC_embeds, DNA_embeds, siglip, args.avg_hic_emb)

                if args.evo2_option == 'Yes':
                    DNA_row_embeds = DNA_row_embeds.transpose(-1, -2)
                    DNA_row_embeds = cut1d(DNA_row_embeds, boundary)
                    DNA_row_embeds = projection0(DNA_row_embeds)

                    evo2_embed1s = cut1d(data['embedding_row'], boundary)
                    evo2_embed1s = projection1(evo2_embed1s)

                    if args.evo2_negative_sample > 0:
                        poses = data['positions'][..., 4:7].flatten(0, 1).cpu()
                        ids = np.random.randint(len(evo2_negative_poses), size = min_candidates)
                        evo2_negative_embeds_no_overlap = []
                        for i in ids:
                            flag = True
                            for j in range(poses.shape[0]):
                                if torch.all(evo2_negative_poses[i] == poses[j]):
                                    flag = False
                                    continue
                            if flag:
                                evo2_negative_embeds_no_overlap.append(evo2_negative_embeds[i].cuda())
                                if len(evo2_negative_embeds_no_overlap) >= args.evo2_negative_sample:
                                    break

                        evo2_negative_embed1s = torch.stack(evo2_negative_embeds_no_overlap, dim=0)
                        evo2_negative_embed1s = projection1(evo2_negative_embed1s)
                    else:
                        evo2_negative_embed1s = None

                    loss1 = siglip_DNA_evo2(DNA_row_embeds, evo2_embed1s, siglip2, evo2_negative_embed1s)
                else:
                    loss1 = torch.tensor(0)

                loss = loss0 + loss1

                accelerator.backward(loss)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step += 1

            if args.evo2_option == 'Yes' and args.evo2_negative_sample > 0:
                # maintaining negative candidates
                for i in range(data['positions'].shape[0]):
                    for j in range(data['positions'].shape[1]):
                        flag = True
                        for neg_pos in evo2_negative_poses:
                            if torch.all(data['positions'][i,j,...,4:7].cpu() == neg_pos):
                                flag = False
                                break
                        if flag:
                            evo2_negative_poses.append(data['positions'][i,j, ..., 4:7].cpu())
                            evo2_negative_embeds.append(cut1d(data['embedding_row'][i,j], boundary).cpu())
                            if len(evo2_negative_embeds) > max_candidates:
                                evo2_negative_poses  = evo2_negative_poses[max_candidates//2:]
                                evo2_negative_embeds = evo2_negative_embeds[max_candidates//2:]

            train_losses.append(loss.cpu().item())

            learning_rate = scheduler.get_last_lr()[0]

            if accelerator.is_main_process and not args.debug:
                results = {
                    'train loss (DNA-HiC)' : loss0.cpu().item(),
                    'train loss (overall)' : loss.cpu().item(),
                    'learning rate' : learning_rate
                }
                if args.evo2_option == 'Yes':
                    results.update(
                        {
                            'train loss (DNA-evo2)' : loss1.cpu().item(),
                        }
                    )
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
        siglip.eval()
        if args.evo2_option == 'Yes':
            siglip2.eval()
            projection0.eval()
            projection1.eval()

        if accelerator.is_main_process:
            ckpt = os.path.join(out_dir, f'{step}.pt')
            state = {
                'model' : accelerator.unwrap_model(model).state_dict(),
                'siglip' : accelerator.unwrap_model(siglip).state_dict(),
                'step' : step,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }
            if args.evo2_option == 'Yes':
                state.update(
                    {
                        'siglip2' : accelerator.unwrap_model(siglip2).state_dict(),
                        'projection0' : accelerator.unwrap_model(projection0).state_dict(),
                        'projection1' : accelerator.unwrap_model(projection1).state_dict(),
                    }
                )

            torch.save(state, ckpt)

            ckpt_maintainer.create_checkpoint(ckpt, keep = (step % 10000 == 0))

            ckpt_maintainer.link_checkpoint(ckpt, os.path.join(out_dir, f'last.pt'))

            print_info(("Step: %d, \t Train, \t LR: %0.8f, \t Loss: %0.8f" % (step, learning_rate, np.mean(train_losses))))

            train_losses = []

        results = {}
        def val(dl, key_suffix = ''):
            with torch.no_grad():
                valid_bar = tqdm(dl, disable = not accelerator.is_main_process)
                valid_loss0es = []
                valid_rank0es = []
                valid_loss1es = []
                valid_losses = []
                first = True
                for data in valid_bar:
                    HiC_embeds, DNA_embeds, DNA_row_embeds = model(**data, return_emb_directly = True)

                    DNA_embeds = cut2d(DNA_embeds, boundary)
                    HiC_embeds = cut2d(HiC_embeds, boundary)

                    loss0, rank0 = siglip_HiC_DNA(HiC_embeds, DNA_embeds, siglip, args.avg_hic_emb)

                    if args.evo2_option == 'Yes':
                        DNA_row_embeds = DNA_row_embeds.transpose(-1, -2)
                        DNA_row_embeds = cut1d(DNA_row_embeds, boundary)
                        DNA_row_embeds = projection0(DNA_row_embeds)

                        evo2_embed1s = cut1d(data['embedding_row'], boundary)
                        evo2_embed1s = projection1(evo2_embed1s)

                        loss1 = siglip_DNA_evo2(DNA_row_embeds, evo2_embed1s, siglip2)
                    else:
                        loss1 = torch.tensor(0)

                    loss = loss0 + loss1

                    valid_losses.append(loss)
                    valid_loss0es.append(loss0)
                    valid_loss1es.append(loss1)

            valid_losses = torch.stack(valid_losses)
            valid_loss0es = torch.stack(valid_loss0es)

            valid_losses = accelerator.gather(valid_losses)
            valid_loss0es = accelerator.gather(valid_loss0es)

            if args.evo2_option == 'Yes':
                valid_loss1es = torch.stack(valid_loss1es)
                valid_loss1es = accelerator.gather(valid_loss1es)

            if accelerator.is_main_process:
                valid_loss = valid_losses.mean().item()
                k = 'valid loss (sum)'
                results[k] = valid_loss
                print_info(f"Step: {step}, \t {k} is {valid_loss}")
                if valid_loss < best.get(k, 100):
                    best[k] = valid_loss
                    ckpt_maintainer.link_checkpoint(ckpt, os.path.join(out_dir, f'best_valid_loss.pt'))

                valid_loss0 = valid_loss0es.mean().item()
                k = 'valid loss (DNA-HiC)'
                results[k] = valid_loss0
                print_info(f"Step: {step}, \t {k} is {valid_loss0}")
                if valid_loss0 < best.get(k, 100):
                    best[k] = valid_loss0
                    ckpt_maintainer.link_checkpoint(ckpt, os.path.join(out_dir, f'best_valid_loss_DNAHiC.pt'))

                if args.evo2_option == 'Yes':
                    valid_loss1 = valid_loss1es.mean().item()
                    k = 'valid loss (DNA-evo2)'
                    results[k] = valid_loss1
                    print_info(f"Step: {step}, \t {k} is {valid_loss1}")
                    if valid_loss1 < best.get(k, 100):
                        best[k] = valid_loss1
                        ckpt_maintainer.link_checkpoint(ckpt, os.path.join(out_dir, f'best_valid_loss_DNAEvo2.pt'))

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
    parser.add_argument('--avg-hic-emb', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--evo2-negative-sample', type=int, default = 256)
    parser.add_argument('--evo2-projection-dim', type=int, default = 512)
    parser.set_defaults(
        method_name = 'CDNA2d-siglip-pretrain',

        resolution = 2000,

        input_option = 'HC',
        target_option = 'hic',

        max_separation = 500000,

        chunk = 100,
        stride = 80,

        evo2_option = 'Yes',

        max_step = 50000,
        batch_size = 3,
        sample = True,
        whole_row = True,
        train_pos_per_row = 2,
        valid_pos_per_row = 2,
        train_hic_per_pos = 8,
        valid_hic_per_pos = 8
    )
    args = parser.parse_args()

    train(args)