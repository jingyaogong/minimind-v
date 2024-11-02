import os
import random
import numpy as np
import torch
import warnings
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model import Transformer
from model.LMConfig import LMConfig
from model.vision_utils import get_vision_model, get_img_process, get_img_embedding

warnings.filterwarnings('ignore')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_model(lm_config, device, multi):
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    model_from = 1  # 1从权重，2用transformers

    if model_from == 1:
        moe_path = '_moe' if lm_config.use_moe else ''
        if multi:
            ckp = f'./out/{lm_config.dim}{moe_path}_vlm_sft_multi.pth'
        else:
            ckp = f'./out/{lm_config.dim}{moe_path}_vlm_sft.pth'
        model = Transformer(lm_config)
        state_dict = torch.load(ckp, map_location=device)

        # 处理不需要的前缀
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict, strict=False)
    else:
        model = AutoModelForCausalLM.from_pretrained('minimind-v-v1-small', trust_remote_code=True)

    model = model.to(device)
    print(f'模型参数: {count_parameters(model) / 1e6} 百万 = {count_parameters(model) / 1e9} B (Billion)')

    vision_model, preprocess = get_vision_model(encoder_type="clip")
    vision_model = vision_model.to(device)
    return model, tokenizer, vision_model, preprocess


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # -------------------------- 基本参数设置 -----------------------------------
    multi = False  # 设置multi参数，控制单图或多图推理
    out_dir = 'out'
    temperature = 0.5
    top_k = 8
    setup_seed(1337)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dtype = 'bfloat16'
    max_seq_len = 1024
    encoder_type="clip"
    # lm_config = LMConfig()
    if encoder_type == "clip":
        lm_config = LMConfig()
    else:
        lm_config = LMConfig(image_special_token='<'*98+'>'*98, image_ids=[30]*98+[32]*98)
    lm_config.max_seq_len = max_seq_len
    model, tokenizer, vision_model, preprocess = init_model(lm_config, device, multi)
    model.eval()

    # -------------------------- 问题和目录设置 -----------------------------------
    if multi:
        image_dir = './dataset/eval_multi_images/bird/'
        prompt = "<image>\n<image>\nName all the differences between these two birds."
    else:
        image_dir = './dataset/eval_images/'
        prompt = lm_config.image_special_token + '\n这个图片描述的是什么内容？'

    image_files = sorted(os.listdir(image_dir))

    # -------------------------- 推理逻辑 -----------------------------------
    if multi:
        # 多图推理：所有图像编码一次性推理
        image_encoders = []
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            image = Image.open(image_path).convert('RGB')
            image_process = get_img_process(image, preprocess).to(vision_model.device)
            image_encoder = get_img_embedding(image_process, vision_model).unsqueeze(0)
            image_encoders.append(image_encoder)
            print(f'[Image]: {image_file}')
        image_encoders = torch.cat(image_encoders, dim=0).unsqueeze(0)

        messages = [{"role": "user", "content": prompt}]
        new_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)[
                     -(max_seq_len - 1):]
        x = tokenizer(new_prompt).data['input_ids']
        x = torch.tensor(x, dtype=torch.long, device=device)[None, ...]

        with torch.no_grad():
            res_y = model.generate(x, tokenizer.eos_token_id, max_new_tokens=max_seq_len, temperature=temperature,
                                   top_k=top_k, stream=True, image_encoders=image_encoders)
            print('[A]: ', end='')
            history_idx = 0
            for y in res_y:
                answer = tokenizer.decode(y[0].tolist())
                if answer and answer[-1] == '�':
                    y = next(res_y)
                    continue
                print(answer[history_idx:], end='', flush=True)
                history_idx = len(answer)
            print('\n')

    else:
        # 单图推理：对每张图像单独推理
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            image = Image.open(image_path).convert('RGB')
            image_process = get_img_process(image, preprocess).to(vision_model.device)
            image_encoder = get_img_embedding(image_process, vision_model).unsqueeze(0)

            messages = [{"role": "user", "content": prompt}]
            new_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)[
                         -(max_seq_len - 1):]
            x = tokenizer(new_prompt).data['input_ids']
            x = torch.tensor(x, dtype=torch.long, device=device)[None, ...]

            print(f'[Image]: {image_file}')
            with torch.no_grad():
                res_y = model.generate(x, tokenizer.eos_token_id, max_new_tokens=max_seq_len, temperature=temperature,
                                       top_k=top_k, stream=True, image_encoders=image_encoder)
                print('[A]: ', end='')
                history_idx = 0
                for y in res_y:
                    answer = tokenizer.decode(y[0].tolist())
                    if answer and answer[-1] == '�':
                        y = next(res_y)
                        continue
                    print(answer[history_idx:], end='', flush=True)
                    history_idx = len(answer)
                print('\n')