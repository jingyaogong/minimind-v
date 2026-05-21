"""
DriveMind-V DPO Evaluation Script
Compares SFT baseline vs DPO-aligned model on driving preference dataset.

Metrics:
  - Preference accuracy (% of pairs where chosen log-prob > rejected)
  - Per-category accuracy breakdown
  - Reward margin (chosen - rejected log-prob)
  - Qualitative generation samples

Usage:
  cd /workspace/minimind-v/minimind-v
  uv run python eval_dpo_driving.py
"""
import os
import sys
import json
import copy
import warnings
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from model.model_vlm import MiniMindVLM, VLMConfig
from trainer.lora_utils import apply_lora, get_lora_config
from trainer.trainer_utils import setup_seed

warnings.filterwarnings('ignore')


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_sft_model(weight_path: str, hidden_size: int, num_layers: int, device: str):
    config = VLMConfig(hidden_size=hidden_size, num_hidden_layers=num_layers)
    model = MiniMindVLM(config, vision_model_path='./model/vision_model/clip-vit-base-patch16')
    weights = torch.load(weight_path, map_location=device)
    model.load_state_dict(weights, strict=False)
    return model.eval().to(device)


def load_dpo_model(sft_model, adapter_dir: str, hidden_size: int,
                   num_layers: int, device: str, lora_r: int, lora_alpha: int):
    from peft import PeftModel
    # Start from a fresh copy of the SFT weights, then attach adapters
    config = VLMConfig(hidden_size=hidden_size, num_hidden_layers=num_layers)
    base = MiniMindVLM(config, vision_model_path='./model/vision_model/clip-vit-base-patch16')
    # Copy weights from already-loaded SFT model
    base.load_state_dict(
        {k: v for k, v in sft_model.state_dict().items()}, strict=False
    )
    base = base.to(device)
    # Attach PEFT adapters
    lora_config = get_lora_config(r=lora_r, lora_alpha=lora_alpha)
    base = apply_lora(base, lora_config, verbose=False)
    from peft import set_peft_model_state_dict
    import safetensors.torch as sf
    adapter_file = os.path.join(adapter_dir, 'adapter_model.safetensors')
    adapter_weights = sf.load_file(adapter_file)
    set_peft_model_state_dict(base, adapter_weights)
    return base.eval().to(device)


def compute_log_probs(model, input_ids, attention_mask, prompt_len, device):
    """
    Returns (sum_logp, mean_logp) over response tokens only.
    mean_logp is length-normalized and used for preference accuracy
    to avoid the bias where shorter rejected responses win by default.
    """
    with torch.no_grad():
        attn_float = attention_mask.float()
        outputs = model(input_ids=input_ids, attention_mask=attn_float)
        logits = outputs.logits  # (B, L, V)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    log_probs_all = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs_all.gather(
        2, shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    response_mask = shift_mask.clone()
    response_start = max(0, prompt_len - 1)
    response_mask[0, :response_start] = 0

    token_log_probs = token_log_probs * response_mask.float()
    n_response_tokens = response_mask.float().sum().item()
    n_response_tokens = max(n_response_tokens, 1)

    sum_logp = token_log_probs.sum().item()
    mean_logp = sum_logp / n_response_tokens
    return sum_logp, mean_logp


def tokenize(text, tokenizer, max_length, device):
    enc = tokenizer(
        text, truncation=True, max_length=max_length,
        return_tensors='pt', add_special_tokens=False,
    )
    return enc['input_ids'].to(device), enc['attention_mask'].to(device)


def generate_response(model, tokenizer, prompt: str, device: str,
                      max_new_tokens: int = 128) -> str:
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    with torch.no_grad():
        output_ids = model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_preferences(model, tokenizer, samples, max_length, device, label):
    per_category = defaultdict(lambda: {'correct': 0, 'total': 0, 'margins': []})
    overall = {'correct': 0, 'total': 0, 'margins': []}

    for sample in samples:
        prompt = sample['prompt']
        chosen = sample['chosen']
        rejected = sample['rejected']
        category = sample.get('category', 'unknown')

        prompt_ids, _ = tokenize(prompt, tokenizer, max_length, device)
        prompt_len = prompt_ids.shape[1]

        chosen_ids, chosen_mask = tokenize(prompt + chosen, tokenizer, max_length, device)
        rejected_ids, rejected_mask = tokenize(prompt + rejected, tokenizer, max_length, device)

        chosen_sum, chosen_mean = compute_log_probs(model, chosen_ids, chosen_mask, prompt_len, device)
        rejected_sum, rejected_mean = compute_log_probs(model, rejected_ids, rejected_mask, prompt_len, device)

        # Use length-normalized log-prob to avoid bias toward shorter rejected responses
        correct = chosen_mean > rejected_mean
        margin = chosen_mean - rejected_mean  # per-token margin

        overall['total'] += 1
        overall['correct'] += int(correct)
        overall['margins'].append(margin)

        per_category[category]['total'] += 1
        per_category[category]['correct'] += int(correct)
        per_category[category]['margins'].append(margin)

    return overall, dict(per_category)


# ── Report printing ────────────────────────────────────────────────────────────

def print_report(sft_overall, sft_cats, dpo_overall, dpo_cats, samples, tokenizer,
                 sft_model, dpo_model, device, args):
    sep = '-' * 68

    print('\n' + '=' * 68)
    print('  DriveMind-V DPO Evaluation Report')
    print('=' * 68)

    # Overall
    sft_acc = sft_overall['correct'] / sft_overall['total']
    dpo_acc = dpo_overall['correct'] / dpo_overall['total']
    sft_avg_margin = sum(sft_overall['margins']) / len(sft_overall['margins'])
    dpo_avg_margin = sum(dpo_overall['margins']) / len(dpo_overall['margins'])

    print(f'\n{"Model":<20} {"Pref Acc":>10} {"Avg Margin/tok":>16} {"Pairs":>8}')
    print(sep)
    print(f'{"SFT (baseline)":<20} {sft_acc:>9.1%} {sft_avg_margin:>16.4f} {sft_overall["total"]:>8}')
    print(f'{"DPO (aligned)":<20} {dpo_acc:>9.1%} {dpo_avg_margin:>16.4f} {dpo_overall["total"]:>8}')
    delta_acc = dpo_acc - sft_acc
    delta_margin = dpo_avg_margin - sft_avg_margin
    print(f'{"Delta (DPO - SFT)":<20} {delta_acc:>+9.1%} {delta_margin:>+16.4f}')
    print(f'\n  Note: Margin = mean(chosen_logp) - mean(rejected_logp) per response token.')
    print(f'        Length-normalized to remove bias toward shorter rejected responses.')

    # Per-category
    print(f'\n{"Category":<24} {"SFT Acc":>9} {"DPO Acc":>9} {"Delta":>9} {"SFT Margin":>12} {"DPO Margin":>12}')
    print(sep)
    all_cats = sorted(set(list(sft_cats.keys()) + list(dpo_cats.keys())))
    for cat in all_cats:
        s = sft_cats.get(cat, {'correct': 0, 'total': 1, 'margins': [0]})
        d = dpo_cats.get(cat, {'correct': 0, 'total': 1, 'margins': [0]})
        s_acc = s['correct'] / s['total']
        d_acc = d['correct'] / d['total']
        s_margin = sum(s['margins']) / len(s['margins'])
        d_margin = sum(d['margins']) / len(d['margins'])
        delta = d_acc - s_acc
        print(f'{cat:<24} {s_acc:>8.1%} {d_acc:>9.1%} {delta:>+9.1%} {s_margin:>12.2f} {d_margin:>12.2f}')

    # Margin distribution
    print(f'\nMargin Distribution (mean chosen_logp/tok - mean rejected_logp/tok):')
    print(sep)
    for label, overall in [('SFT', sft_overall), ('DPO', dpo_overall)]:
        margins = overall['margins']
        pos = sum(1 for m in margins if m > 0)
        neg = sum(1 for m in margins if m <= 0)
        mn = min(margins)
        mx = max(margins)
        med = sorted(margins)[len(margins) // 2]
        print(f'  {label}: min={mn:.1f}  median={med:.1f}  max={mx:.1f}  pos={pos}  neg={neg}')

    # Qualitative samples
    print(f'\nQualitative Generation Samples (DPO model):')
    print(sep)
    # Pick one example per category (first occurrence)
    seen_cats = set()
    qual_samples = []
    for s in samples:
        cat = s.get('category', 'unknown')
        if cat not in seen_cats:
            seen_cats.add(cat)
            qual_samples.append(s)
        if len(qual_samples) >= 6:
            break

    for s in qual_samples:
        prompt = s['prompt']
        chosen = s['chosen']
        rejected = s['rejected']
        cat = s.get('category', 'unknown')
        # Extract scenario text from prompt
        scenario_start = prompt.find('Scenario:')
        scenario_end = prompt.find('[/INST]')
        scenario = prompt[scenario_start:scenario_end].strip() if scenario_start != -1 else prompt[-120:]

        print(f'\n  [{cat}]')
        print(f'  Scenario: {scenario[10:].strip()}')
        sft_resp = generate_response(sft_model, tokenizer, prompt, device, max_new_tokens=args.max_new_tokens)
        dpo_resp = generate_response(dpo_model, tokenizer, prompt, device, max_new_tokens=args.max_new_tokens)
        print(f'  SFT    : {sft_resp}')
        print(f'  DPO    : {dpo_resp}')
        print(f'  Chosen : {chosen}')
        print(f'  Rejected: {rejected}')

    # Summary
    print(f'\n{"=" * 68}')
    print('  Summary')
    print(f'{"=" * 68}')
    print(f'  Dataset : {sft_overall["total"]} preference pairs across {len(all_cats)} categories')
    print(f'  SFT baseline preference accuracy : {sft_acc:.1%}')
    print(f'  DPO aligned preference accuracy  : {dpo_acc:.1%}')
    print(f'  Accuracy improvement             : {delta_acc:+.1%}')
    print(f'  Avg margin improvement           : {delta_margin:+.2f} log-prob units')
    print()
    relative_gain = (dpo_acc - sft_acc) / max(1 - sft_acc, 1e-6)
    if dpo_acc >= 0.90:
        grade = 'STRONG — DPO above 90% preference accuracy'
    elif dpo_acc >= 0.75:
        grade = 'MODERATE — DPO above 75% preference accuracy'
    elif relative_gain >= 0.50:
        grade = f'MEANINGFUL — {relative_gain:.0%} relative gain over SFT; more epochs or data would push higher'
    else:
        grade = 'WEAK — consider more training epochs or a larger/better preference dataset'
    print(f'  Assessment: {grade}')
    print('=' * 68 + '\n')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='DriveMind-V DPO Evaluation')
    parser.add_argument('--data_path', default='dataset/driving_preferences.jsonl')
    parser.add_argument('--sft_weight', default='out/sft_vlm_512.pth')
    parser.add_argument('--dpo_adapter_dir', default='out/dpo_driving_lora_512_adapters')
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_hidden_layers', type=int, default=8)
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--max_new_tokens', type=int, default=80)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    setup_seed(args.seed)
    device = args.device

    print(f'Device: {device}')
    print(f'Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained('./model')

    # Load samples
    samples = []
    with open(args.data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f'Loaded {len(samples)} preference pairs')

    # Load SFT model
    print(f'Loading SFT model from {args.sft_weight}...')
    sft_model = load_sft_model(args.sft_weight, args.hidden_size, args.num_hidden_layers, device)
    print('  SFT model loaded.')

    # Load DPO model
    print(f'Loading DPO model from {args.dpo_adapter_dir}...')
    dpo_model = load_dpo_model(
        sft_model, args.dpo_adapter_dir, args.hidden_size,
        args.num_hidden_layers, device, args.lora_r, args.lora_alpha
    )
    print('  DPO model loaded.')

    # Evaluate both models
    print('\nEvaluating SFT model...')
    sft_overall, sft_cats = evaluate_preferences(
        sft_model, tokenizer, samples, args.max_length, device, 'SFT'
    )

    print('Evaluating DPO model...')
    dpo_overall, dpo_cats = evaluate_preferences(
        dpo_model, tokenizer, samples, args.max_length, device, 'DPO'
    )

    print('\nGenerating qualitative samples...')
    print_report(sft_overall, sft_cats, dpo_overall, dpo_cats, samples,
                 tokenizer, sft_model, dpo_model, device, args)


if __name__ == '__main__':
    main()
