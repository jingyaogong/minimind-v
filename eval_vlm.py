import argparse
import os
import random
import numpy as np
import torch
import warnings
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model_vlm import MiniMindVLM
from model.VLMConfig import VLMConfig
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

warnings.filterwarnings('ignore')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_model(lm_config, device):
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    if args.load == 0:
        moe_path = '_moe' if args.use_moe else ''
        modes = {0: 'pretrain_vlm', 1: 'sft_vlm', 2: 'sft_vlm_multi'}
        ckp = f'./{args.out_dir}/{modes[args.model_mode]}_{args.dim}{moe_path}.pth'
        model = MiniMindVLM(lm_config)
        state_dict = torch.load(ckp, map_location=device)
        model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=False)
    else:
        transformers_model_path = 'MiniMind2-V'
        tokenizer = AutoTokenizer.from_pretrained(transformers_model_path)
        model = AutoModelForCausalLM.from_pretrained(transformers_model_path, trust_remote_code=True)

    print(f'VLM参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')

    vision_model, preprocess = MiniMindVLM.get_vision_model()
    return model.eval().to(device), tokenizer, vision_model.eval().to(device), preprocess


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
    parser.add_argument('--max_seq_len', default=8192, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    # 默认单图推理，设置为2为多图推理
    parser.add_argument('--use_multi', default=1, type=int)
    parser.add_argument('--stream', default=True, type=bool)
    parser.add_argument('--load', default=0, type=int, help="0: 原生torch权重，1: transformers加载")
    parser.add_argument('--model_mode', default=1, type=int,
                        help="0: Pretrain模型，1: SFT模型，2: SFT-多图模型 (beta拓展)")
    args = parser.parse_args()

    lm_config = VLMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)

    model, tokenizer, vision_model, preprocess = init_model(lm_config, args.device)


    def chat_with_vlm(prompt, pixel_tensors, image_names):
        messages = [{"role": "user", "content": prompt}]

        new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )[-args.max_seq_len + 1:]

        print(f'[Image]: {image_names}')
        with torch.no_grad():
            x = torch.tensor(tokenizer(new_prompt)['input_ids'], device=args.device).unsqueeze(0)
            outputs = model.generate(
                x,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_seq_len,
                temperature=args.temperature,
                top_p=args.top_p,
                stream=True,
                pad_token_id=tokenizer.pad_token_id,
                pixel_tensors=pixel_tensors
            )
            print('🤖️: ', end='')
            try:
                if not args.stream:
                    print(tokenizer.decode(outputs.squeeze()[x.shape[1]:].tolist(), skip_special_tokens=True), end='')
                else:
                    history_idx = 0
                    for y in outputs:
                        answer = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
                        if (answer and answer[-1] == '�') or not answer:
                            continue
                        print(answer[history_idx:], end='', flush=True)
                        history_idx = len(answer)
            except StopIteration:
                print("No answer")
            print('\n')


    # 单图推理：每1个图像单独推理
    if args.use_multi == 1:
        image_dir = './dataset/eval_images/'
        prompt = f"{model.params.image_special_token}\n描述一下这个图像的内容。"

        for image_file in os.listdir(image_dir):
            image = Image.open(os.path.join(image_dir, image_file)).convert('RGB')
            pixel_tensors = MiniMindVLM.image2tensor(image, preprocess).to(args.device).unsqueeze(0)
            chat_with_vlm(prompt, pixel_tensors, image_file)

    # 2图推理：目录下的两个图像编码，一次性推理（power by ）
    if args.use_multi == 2:
        args.model_mode = 2
        image_dir = './dataset/eval_multi_images/bird/'
        prompt = (f"{lm_config.image_special_token}\n"
                  f"{lm_config.image_special_token}\n"
                  f"比较一下两张图像的异同点。")
        pixel_tensors_multi = []
        for image_file in os.listdir(image_dir):
            image = Image.open(os.path.join(image_dir, image_file)).convert('RGB')
            pixel_tensors_multi.append(MiniMindVLM.image2tensor(image, preprocess))
        pixel_tensors = torch.cat(pixel_tensors_multi, dim=0).to(args.device).unsqueeze(0)
        # 同样内容重复10次
        for _ in range(10):
            chat_with_vlm(prompt, pixel_tensors, (', '.join(os.listdir(image_dir))))
