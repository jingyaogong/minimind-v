#!/bin/bash

# 设置运行脚本的参数
python train_pretrain_vlm.py \
    --out_dir="out" \
    --epochs=4 \
    --batch_size=16 \
    --learning_rate=0.0004 \
    --device="cuda:0" \
    --dtype="bfloat16" \
    # --use_wandb \ # 如果使用wandb，请取消注释
    --wandb_project="MiniMind-V" \
    --num_workers=8 \
    --data_path="./dataset/pretrain_data.jsonl" \
    --images_path="./dataset/pretrain_images" \
    # --ddp \ # 如果使用DDP分布式训练，请取消注释
    --accumulation_steps=1 \
    --grad_clip=1.0 \
    --warmup_iters=0 \
    --log_interval=100 \
    --save_interval=100 \
    --local_rank=-1 \
    # 模型参数
    --dim=512 \
    --n_layers=8 \
    --max_seq_len=640 \
    --use_moe=False
