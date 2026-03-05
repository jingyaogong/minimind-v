"""
SFT VLM Training with LoRA Support
Extension of train_sft_vlm.py with LoRA for parameter-efficient fine-tuning
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from model.model_vlm import MiniMindVLM, VLMConfig
from dataset.lm_dataset import VLMDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, init_distributed_mode, setup_seed, init_vlm_model, vlm_checkpoint, SkipBatchSampler
from trainer.lora_utils import get_lora_config, apply_lora

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    for step, (input_ids, labels, pixel_values) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        pixel_values = pixel_values.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels, pixel_values=pixel_values)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            logger.log(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.8f} epoch_time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iters,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iters // 60 - spend_time // 60))

            if (wandb is not None) and is_main_process():
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch": epoch + 1,
                           "step": step})

        if step % args.save_interval == 0 or step == iters - 1:
            if is_main_process():
                vlm_checkpoint(model.module if hasattr(model, 'module') else model,
                             epoch,
                             step,
                             args,
                             suffix=f"_lora" if args.use_lora else "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind VLM SFT with LoRA")
    parser.add_argument("--out_dir", type=str, default="out", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float16")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--num_workers", type=int, default=0, help="Data loader workers")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--warmup_iters", type=int, default=0, help="Warmup iterations")
    parser.add_argument("--log_interval", type=int, default=100, help="Log interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="Save interval")
    parser.add_argument("--local_rank", type=int, default=-1, help="DDP local rank")
    parser.add_argument("--use_moe", action="store_true", help="Use Mixture of Experts")
    parser.add_argument("--resume_step", type=int, default=0, help="Resume from step")
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden size")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of layers")

    # LoRA-specific arguments
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for parameter-efficient training")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    args = parser.parse_args()

    # Setup
    init_distributed_mode(args)
    setup_seed(42)
    logger = Logger(args.out_dir, is_main_process())

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('./model')

    # Create VLM config
    lm_config = VLMConfig(
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_heads,
        num_hidden_layers=args.num_layers,
        use_moe=args.use_moe,
    )

    # Initialize model
    model = init_vlm_model(lm_config, args.device)

    # Apply LoRA if requested
    if args.use_lora:
        print("\n" + "=" * 50)
        print("🚀 Applying LoRA for Parameter-Efficient Training")
        print("=" * 50)

        lora_config = get_lora_config(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )

        model = apply_lora(model, lora_config, verbose=True)
        print("=" * 50 + "\n")

    # DDP if needed
    if args.ddp:
        model = DistributedDataParallel(model, device_ids=[args.local_rank])

    # Optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),  # Only train LoRA params if using LoRA
        lr=args.learning_rate
    )

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    autocast_ctx = (
        torch.cuda.amp.autocast(dtype=torch.bfloat16)
        if args.dtype == 'bfloat16'
        else torch.cuda.amp.autocast(dtype=torch.float16)
        if args.dtype == 'float16'
        else nullcontext()
    )

    # Load dataset
    dataset = VLMDataset(
        parquet_path='./dataset/sft_vlm.parquet',
        tokenizer=tokenizer,
        preprocess=model.processor if not args.ddp else model.module.processor,
        max_length=512
    )

    # Data loader
    if args.ddp:
        sampler = DistributedSampler(dataset)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )

    # WandB
    wandb = None
    if args.use_wandb and is_main_process():
        import wandb as wandb_lib
        wandb = wandb_lib
        wandb.init(
            project="drivemind-v",
            name=f"sft_lora_r{args.lora_r}" if args.use_lora else "sft_full",
            config=vars(args)
        )

    # Training loop
    iters = len(loader)
    logger.log(f"Starting SFT training {'with LoRA' if args.use_lora else 'full fine-tuning'}...")
    logger.log(f"Total iterations: {iters} per epoch")

    for epoch in range(args.epochs):
        train_epoch(epoch, loader, iters, start_step=0, wandb=wandb)

    # Save final model
    if is_main_process():
        if args.use_lora:
            # Save LoRA adapters (tiny file ~2-5MB)
            save_path = f"{args.out_dir}/lora_adapters_sft_vlm_{args.hidden_size}"
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(save_path)
            logger.log(f"✅ LoRA adapters saved to {save_path}")
        else:
            vlm_checkpoint(
                model.module if hasattr(model, 'module') else model,
                args.epochs - 1,
                iters - 1,
                args,
                suffix="_final"
            )

    if wandb:
        wandb.finish()

    logger.log("Training complete!")
