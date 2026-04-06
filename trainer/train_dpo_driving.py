"""
DPO (Direct Preference Optimization) trainer for DriveMind-V.
Trains the model to prefer safe driving actions over risky ones.

Dataset format (JSONL):
  {"prompt": "...", "chosen": "safe action", "rejected": "risky action"}

DPO loss:
  L = -E[log σ(β * (log π(yw|x) - log πref(yw|x)) - β * (log π(yl|x) - log πref(yl|x)))]

Usage:
  uv run python trainer/train_dpo_driving.py \
    --data_path dataset/driving_preferences.jsonl \
    --use_lora --epochs 3
"""
import os
import sys
import copy
import json

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from torch import optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from model.model_vlm import MiniMindVLM, VLMConfig
from trainer.trainer_utils import Logger, setup_seed, get_lr
from trainer.lora_utils import get_lora_config, apply_lora

warnings.filterwarnings('ignore')


# ── Dataset ──────────────────────────────────────────────────────────────────

class DrivingPreferenceDataset(Dataset):
    """
    Loads (prompt, chosen, rejected) pairs and tokenizes them.
    Returns:
        chosen_ids:    [prompt + chosen] token ids
        rejected_ids:  [prompt + rejected] token ids
        prompt_len:    length of prompt tokens (so we can mask the loss)
    """
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        Logger(f"Loaded {len(self.samples)} preference pairs from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def _tokenize(self, text: str):
        return self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            add_special_tokens=False,
        )['input_ids'].squeeze(0)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt = sample['prompt']
        chosen = sample['chosen']
        rejected = sample['rejected']

        prompt_ids = self._tokenize(prompt)
        chosen_ids = self._tokenize(prompt + chosen)
        rejected_ids = self._tokenize(prompt + rejected)

        prompt_len = len(prompt_ids)

        return {
            'chosen_ids': chosen_ids,
            'rejected_ids': rejected_ids,
            'prompt_len': prompt_len,
        }


def collate_fn(batch, pad_id=0):
    """Pad sequences to same length within a batch."""
    chosen_ids = [x['chosen_ids'] for x in batch]
    rejected_ids = [x['rejected_ids'] for x in batch]
    prompt_lens = [x['prompt_len'] for x in batch]

    def pad(seqs):
        max_len = max(len(s) for s in seqs)
        padded = torch.zeros(len(seqs), max_len, dtype=torch.long)
        mask = torch.zeros(len(seqs), max_len, dtype=torch.bool)
        for i, s in enumerate(seqs):
            padded[i, :len(s)] = s
            mask[i, :len(s)] = True
        return padded, mask

    chosen_ids, chosen_mask = pad(chosen_ids)
    rejected_ids, rejected_mask = pad(rejected_ids)
    prompt_lens = torch.tensor(prompt_lens, dtype=torch.long)

    return chosen_ids, chosen_mask, rejected_ids, rejected_mask, prompt_lens


# ── DPO loss ─────────────────────────────────────────────────────────────────

def compute_log_probs(model, input_ids, attention_mask, prompt_len):
    """
    Compute sum of log-probs over response tokens only (not prompt).

    Args:
        model: language model
        input_ids: (B, L)
        attention_mask: (B, L)
        prompt_len: (B,) — number of prompt tokens per sample

    Returns:
        log_probs: (B,) — sum of log-probs over response tokens
    """
    with torch.no_grad() if not model.training else nullcontext():
        # Model expects float attention_mask (0.0/1.0), not bool
        attn_mask_float = attention_mask.float()
        outputs = model(input_ids=input_ids, attention_mask=attn_mask_float)
        logits = outputs.logits  # (B, L, V)

    # Shift: logits[i] predicts input_ids[i+1]
    shift_logits = logits[:, :-1, :].contiguous()          # (B, L-1, V)
    shift_labels = input_ids[:, 1:].contiguous()            # (B, L-1)
    shift_mask = attention_mask[:, 1:].contiguous()         # (B, L-1)

    log_probs_all = F.log_softmax(shift_logits, dim=-1)     # (B, L-1, V)
    # Gather log-prob of the actual next token
    token_log_probs = log_probs_all.gather(
        2, shift_labels.unsqueeze(-1)
    ).squeeze(-1)                                            # (B, L-1)

    # Mask: only response tokens (after prompt)
    response_mask = shift_mask.clone()
    for i in range(len(prompt_len)):
        # prompt_len[i]-1 because we shifted by 1
        response_start = max(0, prompt_len[i].item() - 1)
        response_mask[i, :response_start] = 0

    # Sum log-probs over response tokens
    token_log_probs = token_log_probs * response_mask.float()
    log_probs = token_log_probs.sum(dim=-1)                  # (B,)

    return log_probs


def dpo_loss(policy_chosen_logp, policy_rejected_logp,
             ref_chosen_logp, ref_rejected_logp, beta: float = 0.1):
    """
    DPO loss from Rafailov et al. (2023).

    L = -E[log σ(β * ((log π(yw|x) - log πref(yw|x)) - (log π(yl|x) - log πref(yl|x))))]
    """
    chosen_rewards = beta * (policy_chosen_logp - ref_chosen_logp)
    rejected_rewards = beta * (policy_rejected_logp - ref_rejected_logp)
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

    # Preference accuracy: fraction of samples where chosen > rejected
    with torch.no_grad():
        pref_acc = (chosen_rewards > rejected_rewards).float().mean()

    return loss, chosen_rewards.mean().detach(), rejected_rewards.mean().detach(), pref_acc


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(epoch, loader, iters, policy_model, ref_model, optimizer,
                scaler, autocast_ctx, args, wandb=None):
    policy_model.train()
    start_time = time.time()
    total_loss = 0.0
    total_acc = 0.0

    for step, (chosen_ids, chosen_mask, rejected_ids, rejected_mask, prompt_lens) in \
            enumerate(loader, start=1):

        chosen_ids = chosen_ids.to(args.device)
        chosen_mask = chosen_mask.to(args.device)
        rejected_ids = rejected_ids.to(args.device)
        rejected_mask = rejected_mask.to(args.device)
        prompt_lens = prompt_lens.to(args.device)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        with autocast_ctx:
            # Policy log-probs (with gradient)
            policy_chosen_logp = compute_log_probs(
                policy_model, chosen_ids, chosen_mask, prompt_lens)
            policy_rejected_logp = compute_log_probs(
                policy_model, rejected_ids, rejected_mask, prompt_lens)

            # Reference log-probs (no gradient)
            with torch.no_grad():
                ref_chosen_logp = compute_log_probs(
                    ref_model, chosen_ids, chosen_mask, prompt_lens)
                ref_rejected_logp = compute_log_probs(
                    ref_model, rejected_ids, rejected_mask, prompt_lens)

            loss, chosen_r, rejected_r, pref_acc = dpo_loss(
                policy_chosen_logp, policy_rejected_logp,
                ref_chosen_logp, ref_rejected_logp,
                beta=args.beta,
            )
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * args.accumulation_steps
        total_acc += pref_acc.item()

        if step % args.log_interval == 0 or step == iters:
            elapsed = time.time() - start_time
            avg_loss = total_loss / step
            avg_acc = total_acc / step
            Logger(
                f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) '
                f'loss:{avg_loss:.4f} pref_acc:{avg_acc:.3f} '
                f'chosen_r:{chosen_r:.3f} rejected_r:{rejected_r:.3f} '
                f'lr:{lr:.2e} time:{elapsed/60:.1f}min'
            )
            if wandb:
                wandb.log({
                    'loss': avg_loss,
                    'preference_accuracy': avg_acc,
                    'chosen_reward': chosen_r.item(),
                    'rejected_reward': rejected_r.item(),
                    'lr': lr,
                    'epoch': epoch + 1,
                })

    return total_loss / iters, total_acc / iters


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DriveMind-V DPO Training')
    parser.add_argument('--data_path', type=str, default='dataset/driving_preferences.jsonl')
    parser.add_argument('--out_dir', type=str, default='out')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--beta', type=float, default=0.1,
                        help='DPO temperature (higher = closer to reference)')
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--device', type=str,
                        default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', type=str,
                        default='bfloat16' if torch.cuda.is_bf16_supported() else 'float16')
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_hidden_layers', type=int, default=8)
    parser.add_argument('--from_weight', type=str, default='sft_vlm',
                        help='Base checkpoint prefix (loads out/<from_weight>_<hidden_size>.pth)')
    # LoRA
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    # WandB
    parser.add_argument('--use_wandb', action='store_true')

    args = parser.parse_args()
    setup_seed(42)

    # ── Tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained('./model')

    # ── Model config ──
    vlm_config = VLMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
    )

    # ── Load policy model ──
    weight_path = f'{args.out_dir}/{args.from_weight}_{vlm_config.hidden_size}.pth'
    Logger(f'Loading base model from {weight_path}')

    policy_model = MiniMindVLM(vlm_config,
                               vision_model_path='./model/vision_model/clip-vit-base-patch16')
    if os.path.exists(weight_path):
        weights = torch.load(weight_path, map_location=args.device)
        policy_model.load_state_dict(weights, strict=False)
        Logger(f'Loaded weights from {weight_path}')
    else:
        Logger(f'WARNING: {weight_path} not found — using random init')

    policy_model = policy_model.to(args.device)

    # ── Frozen reference model (deep copy before LoRA) ──
    Logger('Creating frozen reference model...')
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    Logger('Reference model frozen.')

    # ── Apply LoRA to policy model ──
    if args.use_lora:
        Logger('\nApplying LoRA to policy model...')
        lora_config = get_lora_config(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        policy_model = apply_lora(policy_model, lora_config, verbose=True)

    # ── Dataset & loader ──
    dataset = DrivingPreferenceDataset(args.data_path, tokenizer, args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(args.device != 'cpu'),
    )

    # ── Optimizer & AMP ──
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, policy_model.parameters()),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    if args.dtype == 'bfloat16':
        autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
    elif args.dtype == 'float16':
        autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
    else:
        autocast_ctx = nullcontext()

    # ── WandB ──
    wandb = None
    if args.use_wandb:
        import wandb as wb
        wb.init(project='drivemind-v', name=f'dpo_lora_r{args.lora_r}_beta{args.beta}',
                config=vars(args))
        wandb = wb

    # ── Train ──
    iters = len(loader)
    Logger(f'\nStarting DPO training: {args.epochs} epochs x {iters} steps')
    Logger(f'Dataset: {len(dataset)} pairs | Batch: {args.batch_size} | Beta: {args.beta}')
    Logger(f'LoRA: {args.use_lora} | Device: {args.device} | dtype: {args.dtype}\n')

    for epoch in range(args.epochs):
        avg_loss, avg_acc = train_epoch(
            epoch, loader, iters,
            policy_model, ref_model,
            optimizer, scaler, autocast_ctx,
            args, wandb,
        )
        Logger(f'Epoch {epoch+1} complete — avg_loss: {avg_loss:.4f}  avg_pref_acc: {avg_acc:.3f}')

    # ── Save ──
    os.makedirs(args.out_dir, exist_ok=True)
    lora_tag = '_lora' if args.use_lora else ''
    save_path = f'{args.out_dir}/dpo_driving{lora_tag}_{vlm_config.hidden_size}'

    if args.use_lora:
        policy_model.save_pretrained(save_path + '_adapters')
        Logger(f'LoRA adapters saved -> {save_path}_adapters')
    else:
        raw = getattr(policy_model, '_orig_mod', policy_model)
        state_dict = {k: v.half().cpu() for k, v in raw.state_dict().items()
                      if not k.startswith('vision_encoder.')}
        ckp = save_path + '.pth'
        torch.save(state_dict, ckp)
        Logger(f'Model saved -> {ckp}')

    if wandb:
        wandb.finish()

    Logger('\nDPO training complete!')
