import os
import random
import time

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


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    model_from = 1  # 1从权重，2用transformers

    if model_from == 1:
        moe_path = '_moe' if lm_config.use_moe else ''
        ckp = f'./out/{lm_config.dim}{moe_path}_vlm_sft_multi.pth'

        model = Transformer(lm_config)
        state_dict = torch.load(ckp, map_location=device)

        # 处理不需要的前缀
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        # 加载到模型中
        model.load_state_dict(state_dict, strict=False)
    else:
        model = AutoModelForCausalLM.from_pretrained('minimind-v-v1-small',
                                                     trust_remote_code=True)

    model = model.to(device)

    print(f'模型参数: {count_parameters(model) / 1e6} 百万 = {count_parameters(model) / 1e9} B (Billion)')

    (vision_model, preprocess) = get_vision_model()
    vision_model = vision_model.to(device)
    return model, tokenizer, (vision_model, preprocess)


def setup_seed(seed):
    random.seed(seed)  # 设置 Python 的随机种子
    np.random.seed(seed)  # 设置 NumPy 的随机种子
    torch.manual_seed(seed)  # 设置 PyTorch 的随机种子
    torch.cuda.manual_seed(seed)  # 为当前 GPU 设置随机种子（如果有）
    torch.cuda.manual_seed_all(seed)  # 为所有 GPU 设置随机种子（如果有）
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # 关闭 cuDNN 的自动调优，避免不确定性


if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    out_dir = 'out'
    start = ""
    temperature = 0.5
    top_k = 8
    setup_seed(1337)
    # device = 'cpu'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dtype = 'bfloat16'
    max_seq_len = 1 * 1024
    lm_config = LMConfig()
    lm_config.max_seq_len = max_seq_len
    # -----------------------------------------------------------------------------

    model, tokenizer, (vision_model, preprocess) = init_model(lm_config)

    model = model.eval()
    # 推送到huggingface
    # model.push_to_hub("minimind")
    # tokenizer.push_to_hub("minimind")

    stream = True

    # prompt = lm_config.image_special_token + '\n这个图片描述的是什么内容？'

    i = 0
    # 获取图像文件列表
    image_dir = './dataset/eval_multi_images/'
    image_files = sorted(os.listdir(image_dir))

    image_count = len(image_files)
    special_tokens = lm_config.image_special_token * image_count  # 根据图片数量重复特殊标记

    # 构建提示
    prompt = f"{special_tokens}\n这几张图片是否是同一个类别？"

    # 遍历图像文件
    image_encoders = []
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file)

        # 打开图像并转换为RGB格式
        image = Image.open(image_path).convert('RGB')
        image_process = get_img_process(image, preprocess).to(vision_model.device)

        # 对图像进行编码
        image_encoder = get_img_embedding(image_process, vision_model).unsqueeze(0)
        print(image_encoder.shape)
        image_encoders.append(image_encoder)
        print(f'[Image]: {image_file}')
    image_encoders = torch.cat(image_encoders, dim=0).unsqueeze(0)
    print(image_encoders.shape)

    # 准备问题和消息
    prompt_ = prompt.replace("\n", "")
    print(f'[Q]: {prompt_}')

    messages = [{"role": "user", "content": prompt}]

    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )[-(max_seq_len - 1):]

    x = tokenizer(new_prompt).data['input_ids']
    x = (torch.tensor(x, dtype=torch.long, device=device)[None, ...])

    with torch.no_grad():
        res_y = model.generate(x, tokenizer.eos_token_id, max_new_tokens=max_seq_len, temperature=temperature,
                                top_k=top_k, stream=stream, image_encoders=image_encoders)

        print('[A]: ', end='')
        try:
            y = next(res_y)
        except StopIteration:
            print("No answer")
            exit()

        history_idx = 0
        while y is not None:
            answer = tokenizer.decode(y[0].tolist())
            if answer and answer[-1] == '�':
                try:
                    y = next(res_y)
                except StopIteration:
                    break
                continue
            if not len(answer):
                try:
                    y = next(res_y)
                except StopIteration:
                    break
                continue

            print(answer[history_idx:], end='', flush=True)
            try:
                y = next(res_y)
            except StopIteration:
                break
            history_idx = len(answer)

            if not stream:
                break

        print('\n')