import argparse
import copy
import logging
import math
import os
from pathlib import Path
import json
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
import kornia.augmentation as K
import wandb

from typing import Dict
from model import models_mae_DEX
from utils.dataloader_MedVerse import MedVerseDataset
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger

def adjust_learning_rate(optimizer, init_lr, epoch, args):
    if epoch < args.warmup_epochs:
        cur_lr = init_lr * epoch / args.warmup_epochs
    else:
        cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr

def cosine_momentum(epoch, max_epoch, init_m=0.99, final_m=0.999):
    """Adjust momentum based on current epoch"""
    m = final_m + 0.5 * (init_m - final_m) * (1. + math.cos(math.pi * epoch / max_epoch))
    return m

def cosine_noise_std(epoch, max_epoch, init_std=1.0, final_std=0.01):
    """Cosine decay schedule for noise_std"""
    return final_std + 0.5 * (init_std - final_std) * (1. + math.cos(math.pi * epoch / max_epoch))

def cosine_coef(epoch, max_epoch, init_coef=1.0, final_coef=0.01):
    """Cosine decay schedule for coef"""
    return final_coef + 0.5 * (init_coef - final_coef) * (1. + math.cos(math.pi * epoch / max_epoch))

def collect_actor_activation_counts(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """
    Throughout model, collect all DEX_layer's actor_activation_count
    
    Args:
        model: nn.Module
    
    Returns:
        dict: key is the layer name, value is the actor_activation_count (Tensor)
    """
    counts = {}
    
    for name, module in model.named_modules():
        # Check if the module has the actor_activation_count attribute
        if hasattr(module, "actor_activation_count"):
            counts[name] = module.actor_activation_count.clone().detach()
    
    return counts

def log_actor_activation_counts(counts: Dict[str, torch.Tensor], step: int = None):
    """
    Print or log actor activation counts
    
    Args:
        counts: Dictionary returned by collect_actor_activation_counts
        step: Optional, training step
    """
    prefix = f"[Step {step}] " if step is not None else ""
    for layer_name, tensor in counts.items():
        print(f"{prefix}Layer {layer_name}: {tensor.cpu().numpy()}")

##################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")

    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Create model:
    model = models_mae_DEX.mae_DEX_vit_base(norm_pix_loss=True, img_size=args.resolution)

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    init_lr = args.lr * args.batch_size / 512.
    optimizer = torch.optim.AdamW(model.parameters(), init_lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    # Setup data:
    train_dataset = MedVerseDataset(args.data_dir, args.resolution)
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(train_dataset, batch_size=local_batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    
    # FLOPs and Params
    from fvcore.nn import FlopCountAnalysis, parameter_count
    v = next(train_dataloader.__iter__())
    FLOPs = FlopCountAnalysis(model, v[0:1]).total() / 1e9
    Params = parameter_count(model)[''] / 1e6
    print(f"FLOPs:  {FLOPs:.4f} B")
    print(f"Params: {Params:.4f} M\n")

    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")
    
    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance

    # resume:
    global_step = 0
    start_epoch = 0
    if args.resume_step > 0:
        # if accelerator.is_main_process:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu', weights_only=False
            )
        model.load_state_dict(ckpt['model'], strict=False)
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']
        start_epoch = global_step // (train_dataset.__len__()//local_batch_size)

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(project_name=args.exp_name, config=tracker_config, init_kwargs={"wandb": {"name": f"{args.exp_name}"}},)
        
    progress_bar = tqdm(
        range(0, args.epochs*(train_dataset.__len__()//local_batch_size)),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    app_aug = K.AugmentationSequential(K.RandomHorizontalFlip(),
                                       K.RandomAffine(degrees=15),
                                       K.Normalize(mean=MEAN, std=STD),
                                       data_keys=["input"]).to(device)

    iters_per_epoch = len(train_dataloader)
    
    m = args.init_m

    #----------- Train Loop -----------
    for epoch in range(start_epoch, args.epochs):
        model.train()
        
        counts = collect_actor_activation_counts(model)
        log_actor_activation_counts(counts, step=global_step)

        for i, (images) in enumerate(train_dataloader):
            adjust_learning_rate(optimizer, init_lr, epoch + i / iters_per_epoch, args)
            noise_std = cosine_noise_std(epoch + i / iters_per_epoch, args.epochs, init_std=args.noise_std, final_std=0.)

            if args.m_cos:
                m = cosine_momentum(epoch + i / iters_per_epoch, args.epochs, init_m=args.init_m, final_m=args.final_m)

            images = images.to(device, non_blocking=True)
            im = app_aug(images)

            with accelerator.accumulate(model):
                # Compute output
                loss_mae, co_losses, bal_losses, pred, mask = model(im, mask_ratio=args.mask_ratio, m=m, noise_std=noise_std)
                # unused_loss = sum((p**2).sum() for p in model.parameters() if p.grad is None) * 1e-6

                loss = co_losses * args.co_coeff + bal_losses * args.bal_coeff + loss_mae# + unused_loss

                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # check for non-finite loss / grads
                bad = False
                if not torch.isfinite(loss):
                    bad = True

                if not bad:
                    for p in model.parameters():
                        if p.grad is not None:
                            # any NaN/Inf in gradients -> skip
                            if not torch.isfinite(p.grad).all():
                                bad = True
                                break

                if bad:
                    if accelerator.is_main_process:
                        print(f"[Step {global_step}] Non-finite detected (loss={loss.item() if torch.is_tensor(loss) and torch.isfinite(loss) else loss}), grad_norm={grad_norm}. Skipping step.")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
            if global_step % (args.checkpointing_epochs*(train_dataset.__len__()//local_batch_size)) == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": accelerator.unwrap_model(model).state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")


            logs = {
                "loss_mae": accelerator.gather(loss_mae).mean().detach().item(),
                "loss_co": accelerator.gather(co_losses).mean().detach().item(),
                "loss_bal": accelerator.gather(bal_losses).mean().detach().item(),
                "noise_std": noise_std,
                "m": m,
                "lr": optimizer.param_groups[0]['lr'],
                "grad_norm": accelerator.gather(grad_norm).mean().detach().item()
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)


    model.eval()  # important! This disables randomized embedding dropout
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="/media/volume/LargeDataset/exps", help="path to save logs and models")
    parser.add_argument("--exp-name", type=str, default="MAE_DEX_MedVerse", help="experiment name")
    parser.add_argument("--logging-dir", type=str, default="logs", help="logging directory")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--resume-step", type=int, default=0, help="resume from checkpoint")

    # dataset
    parser.add_argument("--data-dir", type=str, default="/media/volume/LargeDataset/pre-training", help="path to dataset")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256, help="input resolution of images")
    parser.add_argument("--batch-size", type=int, default=896, help="batch size")

    # precision
    parser.add_argument("--allow-tf32", default=True, action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--checkpointing-epochs", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", "--learning-rate", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-6, type=float, metavar="W", help="weight decay (default: 1e-6)")
    parser.add_argument("--max-grad-norm", default=1., type=float, help="Max gradient norm.")
    parser.add_argument('--warmup-epochs', default=10, type=int, metavar='N', help='number of warmup epochs')
    parser.add_argument('--mask-ratio', default=0.75, type=float, metavar='N', help='masking ratio (percentage of removed patches).')
    parser.add_argument('--init-m', default=0.99, type=float, help='updating momentum encoder (default: 0.99)')
    parser.add_argument('--final-m', default=0.999, type=float, help='updating momentum encoder (default: 0.999)')
    parser.add_argument('--m-cos', default=True, type=bool, help='gradually increase momentum to 1 with a half-cycle cosine schedule')
    parser.add_argument('--noise-std', default=1., type=float, help='the init std of noise in router training')

    # seed
    parser.add_argument("--seed", type=int, default=3407)

    # cpu
    parser.add_argument("--num-workers", type=int, default=32)

    # loss
    parser.add_argument("--bal-coeff", type=float, default=0.01)
    parser.add_argument("--co-coeff", type=float, default=0.01)
    
    #
    parser.add_argument("--wandbname", type=str, default="", help='the login name of wandb')

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args

if __name__ == "__main__":
    args = parse_args()
    wandb.login(key=args.wandbname)
    logger = get_logger(__name__)
    main(args)
