import os
import sys
import json

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import transformers
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
from model.model_vlm import MiniMindVLM, VLMConfig

warnings.filterwarnings('ignore', category=UserWarning)


def convert_torch2transformers_minimind(torch_path, transformers_path, dtype=torch.bfloat16):
    VLMConfig.register_for_auto_class()
    MiniMindVLM.register_for_auto_class("AutoModelForCausalLM")
    lm_model = MiniMindVLM(lm_config, vision_model_path="../model/siglip2-base-p16-256-ve")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    lm_model = lm_model.to(dtype)  # 转换模型权重精度
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    del lm_model.vision_encoder
    lm_model.save_pretrained(transformers_path, safe_serialization=False)
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)
    # 显式写入 tie_word_embeddings（save_pretrained 默认不序列化与默认值相同的字段）
    config_path = os.path.join(transformers_path, "config.json")
    config = json.load(open(config_path, 'r', encoding='utf-8'))
    config['tie_word_embeddings'] = True
    # ======= transformers-5.0的兼容低版本写法 =======
    if int(transformers.__version__.split('.')[0]) >= 5:
        tokenizer_config_path = os.path.join(transformers_path, "tokenizer_config.json")
        json.dump({**json.load(open(tokenizer_config_path, 'r', encoding='utf-8')), "tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {}}, open(tokenizer_config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        config['rope_theta'] = lm_config.rope_theta; config['rope_scaling'] = None; config.pop('rope_parameters', None)
    json.dump(config, open(config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"模型已保存为 Transformers-MiniMind-V 格式: {transformers_path}")


def convert_transformers2torch(transformers_path, torch_path):
    model = AutoModelForCausalLM.from_pretrained(transformers_path, trust_remote_code=True)
    torch.save(model.state_dict(), torch_path)
    print(f"模型已保存为 PyTorch 格式: {torch_path}")


if __name__ == '__main__':
    lm_config = VLMConfig(hidden_size=768, num_hidden_layers=8, max_seq_len=8192, use_moe=False)
    torch_path = f"../out/sft_vlm_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    transformers_path = '../minimind-3v'
    convert_torch2transformers_minimind(torch_path, transformers_path)
