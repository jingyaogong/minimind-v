import os
import sys

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import torch
import warnings
import gradio as gr
from queue import Queue
from threading import Thread
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_vlm import MiniMindVLM, VLMConfig
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()
warnings.filterwarnings('ignore')


def scan_vlm_models(base_dir):
    models = {}
    base_dir = os.path.abspath(base_dir)
    for d in sorted(os.listdir(base_dir), reverse=True):
        full_path = os.path.join(base_dir, d)
        if not os.path.isdir(full_path) or d.startswith('.') or d.startswith('_'):
            continue
        files = os.listdir(full_path)
        has_model = any(f.endswith(('.bin', '.safetensors')) for f in files) or 'model.safetensors.index.json' in files
        if has_model:
            models[d] = full_path
    return models


def load_vlm_model(model_path):
    global model, tokenizer, preprocess, lm_config, current_model_name
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model.vision_encoder, model.processor = MiniMindVLM.get_vision_model(vision_model_path)
    if model.vision_encoder is None: raise FileNotFoundError(f"视觉编码器未找到: {vision_model_path}")
    preprocess = model.processor
    lm_config = model.config
    model = model.eval().to(device)
    model.vision_encoder = model.vision_encoder.to(device)
    current_model_name = os.path.basename(model_path)
    param_str = f'{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M'
    print(f'已加载 {current_model_name}，参数量：{param_str}')
    return f"已加载: {current_model_name} ({param_str})"


class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer, queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue
        self.tokenizer = tokenizer

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.queue.put(text)
        if stream_end:
            self.queue.put(None)


def chat(prompt, current_image_path=None):
    global temperature, top_p
    pixel_values = None
    if current_image_path:
        image = Image.open(current_image_path).convert('RGB')
        pixel_values = {k: v.to(model.device) for k, v in MiniMindVLM.image2tensor(image, preprocess).items()}
        prompt = f'{lm_config.image_special_token * lm_config.image_token_len}\n{prompt}'
    messages = [{"role": "user", "content": prompt}]

    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )[-max_seq_len + 1:]

    with torch.no_grad():
        inputs = tokenizer(
            new_prompt,
            return_tensors="pt",
            truncation=True
        ).to(device)
        queue = Queue()
        streamer = CustomStreamer(tokenizer, queue)

        def _generate():
            model.generate(
                inputs.input_ids,
                max_new_tokens=max_seq_len,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                attention_mask=inputs.attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer,
                pixel_values=pixel_values
            )

        Thread(target=_generate).start()

        while True:
            text = queue.get()
            if text is None:
                break
            yield text


def launch_gradio_server(server_name="0.0.0.0", server_port=7788):
    global temperature, top_p
    temperature = args.temperature
    top_p = args.top_p

    def respond(message, history):
        if not message or not message.get("text"):
            yield history + [{"role": "assistant", "content": "请输入问题"}]
            return
        files = message.get("files", [])
        img_path = files[0] if files else None
        text = message["text"]
        if img_path:
            history = history + [{"role": "user", "content": {"path": img_path}}, {"role": "user", "content": text}]
        else:
            history = history + [{"role": "user", "content": text}]
        response = ''
        for chunk in chat(text, img_path):
            response += chunk
            yield history + [{"role": "assistant", "content": response}]

    with gr.Blocks(title="minimind-3v", css="#chatbox img{max-width:120px!important;max-height:120px!important;border-radius:8px} div.full-container{display:flex!important;flex-direction:row!important;flex-wrap:nowrap!important;align-items:center!important} div.full-container>.thumbnails{flex-shrink:0!important;max-width:50px!important} div.full-container>.input-container{flex:1!important} .input-wrapper{display:flex!important;flex-direction:row!important;flex-wrap:nowrap!important;align-items:center!important} .input-wrapper>.thumbnails{flex-shrink:0!important;max-width:50px!important} .input-wrapper>.input-row{flex:1!important} .thumbnail-image{max-width:40px!important;max-height:40px!important} textarea{overflow-y:hidden!important}") as demo:
        gr.HTML('<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;margin:-8px 0 -6px 0"><div style="display:flex;align-items:center;justify-content:center;gap:8px"><img src="https://www.modelscope.cn/api/v1/studio/gongjy/MiniMind/repo?Revision=master&FilePath=images%2Flogo2.png&View=true" style="height:30px"><span style="font-size:1.5rem;font-weight:bold;font-style:italic">minimind-3v</span></div><div style="color:#bbb;font-style:italic;margin:0">AI-generated content may be inaccurate, please verify</div></div>')
        try:
            chatbot = gr.Chatbot(label="", height=520, elem_id="chatbox", type="messages")
        except TypeError:
            chatbot = gr.Chatbot(label="", height=520, elem_id="chatbox")
        msg = gr.MultimodalTextbox(placeholder="输入问题，点击📎上传图片", show_label=False, submit_btn="发送")
        with gr.Row():
            model_dropdown = gr.Dropdown(choices=list(model_dict.keys()), value=current_model_name, show_label=False, scale=1)
            status_text = gr.Textbox(value=f"已加载: {current_model_name}", show_label=False, interactive=False, scale=2)

        def on_model_change(name):
            return load_vlm_model(model_dict[name])

        model_dropdown.change(on_model_change, [model_dropdown], [status_text])
        msg.submit(respond, [msg, chatbot], chatbot)
        demo.launch(server_name=server_name, server_port=server_port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chat with MiniMind-V")
    parser.add_argument('--load_from', default='./', type=str, help="transformers模型扫描目录")
    parser.add_argument('--vision_model', default='../model/siglip2-base-p16-ve', type=str, help="视觉编码器路径")
    parser.add_argument('--temperature', default=0.8, type=float, help="生成温度")
    parser.add_argument('--top_p', default=0.9, type=float, help="nucleus采样阈值")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument('--max_seq_len', default=8192, type=int, help="最大序列长度")
    args = parser.parse_args()

    device = args.device
    max_seq_len = args.max_seq_len
    vision_model_path = args.vision_model
    model_dict = scan_vlm_models(args.load_from)
    if not model_dict:
        print(f"未在 {os.path.abspath(args.load_from)} 找到transformers模型")
        exit(1)
    current_model_name = list(model_dict.keys())[0]
    load_vlm_model(model_dict[current_model_name])
    launch_gradio_server(server_name="0.0.0.0", server_port=8888)
