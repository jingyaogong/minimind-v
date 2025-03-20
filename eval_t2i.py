import argparse
import os
import random
import numpy as np
import torch
import warnings
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model_t2i import MiniMindT2I
from model.VLMConfig import VLMConfig
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

warnings.filterwarnings('ignore')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_model(lm_config, device):
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    moe_path = '_moe' if args.use_moe else ''
    ckp = f'./{args.out_dir}/sft_t2i_{args.dim}{moe_path}.pth'
    model = MiniMindT2I(lm_config)
    state_dict = torch.load(ckp, map_location=device)
    model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=False)

    print(f'T2I参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')

    return model.eval().to(device), tokenizer


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with MiniMind")
    parser.add_argument('--lora_name', default='None', type=str)
    parser.add_argument('--out_dir', default='out', type=str)
    parser.add_argument('--temperature', default=0.65, type=float)
    parser.add_argument('--top_p', default=0.85, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    # MiniMind2-Small (26M)：(dim=512, n_layers=8)
    # MiniMind2 (104M)：(dim=768, n_layers=16)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=640, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    args = parser.parse_args()

    lm_config = VLMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)

    model, tokenizer = init_model(lm_config, args.device)


    def chat_with_vlm(prompt):
        messages = [{"role": "user", "content": prompt}]

        new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )[-args.max_seq_len + 1:]

        with torch.no_grad():
            x = torch.tensor(tokenizer(new_prompt)['input_ids'], device=args.device).unsqueeze(0)
            outputs = model.generate(
                x,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
            image = model.image_tokenizer.decode_code(outputs, (1, 8, 16, 16))
            image = F.interpolate(image, size=[256, 256], mode='bicubic').permute(0, 2, 3, 1)[0]
            image = torch.clamp(127.5 * image + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()
            Image.fromarray(image).save('output.jpg')
            print('🤖️: 图片保存至output.jpg')
            print('\n')


    image_dir = './dataset/eval_images/'
    prompt = f"一个年轻人准备踢足球"
    chat_with_vlm(prompt)