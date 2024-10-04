import gradio as gr
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
    model_from = 2  # 1从权重，2用transformers

    if model_from == 1:
        moe_path = '_moe' if lm_config.use_moe else ''
        ckp = f'./out/{lm_config.dim}{moe_path}_vlm_sft.pth'

        model = Transformer(lm_config)
        state_dict = torch.load(ckp, map_location=device)

        # 处理不需要的前缀
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        for k, v in list(state_dict.items()):
            if 'mask' in k:
                del state_dict[k]

        # 加载到模型中
        model.load_state_dict(state_dict, strict=False)
    else:
        model = AutoModelForCausalLM.from_pretrained('minimind-v-v1',
                                                     trust_remote_code=True)

    model = model.to(device)

    print(f'模型参数: {count_parameters(model) / 1e6} 百万 = {count_parameters(model) / 1e9} B (Billion)')

    (vision_model, preprocess) = get_vision_model()
    vision_model = vision_model.to(device)
    return model, tokenizer, (vision_model, preprocess)


def chat(prompt, current_image_path):
    # 打开图像并转换为RGB格式
    image = Image.open(current_image_path).convert('RGB')
    image_process = get_img_process(image, preprocess).to(vision_model.device)
    # 对图像进行编码
    image_encoder = get_img_embedding(image_process, vision_model).unsqueeze(0)

    prompt = f'{lm_config.image_special_token}\n{prompt}'
    messages = [{"role": "user", "content": prompt}]

    # print(messages)
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )[-(max_seq_len - 1):]

    x = tokenizer(new_prompt).data['input_ids']
    x = (torch.tensor(x, dtype=torch.long, device=device)[None, ...])

    with torch.no_grad():
        res_y = model.generate(x, tokenizer.eos_token_id, max_new_tokens=max_seq_len, temperature=temperature,
                               top_k=top_k, stream=stream, image_encoders=image_encoder)
        # print('[A]: ', end='')
        try:
            y = next(res_y)
        except StopIteration:
            print("No answer")
            return "No answer"

        history_idx = 0
        while y != None:
            answer = tokenizer.decode(y[0].tolist())
            if answer and answer[-1] == '�':
                try:
                    y = next(res_y)
                except:
                    break
                continue
            # print(answer)
            if not len(answer):
                try:
                    y = next(res_y)
                except:
                    break
                continue

            yield answer[history_idx:]
            try:
                y = next(res_y)
            except:
                break
            history_idx = len(answer)
            if not stream:
                break


def launch_gradio_server(server_name="0.0.0.0", server_port=7788):
    with gr.Blocks() as demo:
        # # 顶部下拉菜单
        # model_choice = gr.Dropdown(label="选择模型", choices=["minimind-v1-v", "minimind-v1-small-v"],
        #                            value="minimind-v1-v")
        gr.HTML(f"""
            <div style="text-align: center;">
                <h3>MiniMind-V ({(count_parameters(model) / 1e6):.2f} M) Test Demo 「单轮对话，无历史上下文」</h3>
            </div>
            """)
        # 左侧
        with gr.Row():
            with gr.Column(scale=1):
                def get_current_image_path(image):
                    global current_image_path
                    if image is None:
                        current_image_path = ''
                        return
                    current_image_path = image
                    return current_image_path

                with gr.Blocks() as iface:
                    with gr.Row():
                        image_input = gr.Image(type="filepath", label="选择图片", height=650)
                    image_input.change(fn=get_current_image_path, inputs=image_input)

                def update_parameters(temperature_, top_k_):
                    global temperature, top_k
                    temperature = temperature_
                    top_k = top_k_

                with gr.Blocks() as iface_param:
                    with gr.Row():
                        temperature_slider = gr.Slider(label="Temperature", minimum=0, maximum=1.5, value=0.5)
                        top_k_slider = gr.Slider(label="Top K", minimum=1, maximum=20, value=12)

                    temperature_slider.change(fn=update_parameters, inputs=[temperature_slider, top_k_slider])
                    top_k_slider.change(fn=update_parameters, inputs=[temperature_slider, top_k_slider])

            # 右侧聊天展示和输入框
            with gr.Column(scale=4):
                def chat_with_vlm(message, history):
                    if not message:
                        yield history + [("错误", "错误：提问不能为空。")]
                        return
                    if not current_image_path:
                        yield history + [("错误", "错误：图片不能为空。")]
                        return

                    image_html = f'<img src="gradio_api/file={current_image_path}" alt="Image" style="width:60px;height:auto;">'
                    res_generator = chat(message, current_image_path)
                    response = ''
                    for res in res_generator:
                        response += res
                        # 返回聊天历史，加上新的一轮对话
                        yield history + [(f"{image_html} {message}", response)]

                        # 创建ChatBot组件

                chatbot = gr.Chatbot(label="MiniMind-V Chatbot", height=670)

                # 添加输入框和按钮
                with gr.Row():
                    with gr.Column(scale=8):
                        message_input = gr.Textbox(placeholder="请输入你的问题...", show_label=False, container=False)
                    with gr.Column(scale=1, min_width=50):
                        submit_button = gr.Button("发送")

                # 设置按钮点击事件触发chat_with_vlm函数
                submit_button.click(fn=chat_with_vlm,
                                    inputs=[message_input, chatbot],  # 将消息输入框和聊天历史传入函数
                                    outputs=chatbot)  # 更新Chatbot的输出内容

                # 添加示例问题
                gr.Examples(
                    examples=["描述一下这个图片的内容。", "画面里面有什么？", "画面里的动物在干什么？",
                              "画面里的天气怎么样？"],
                    inputs=message_input)

        demo.launch(server_name=server_name, server_port=server_port)


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # 获取模型配置
    lm_config = LMConfig()
    max_seq_len = lm_config.max_seq_len
    model, tokenizer, (vision_model, preprocess) = init_model(lm_config)
    stream = True
    current_image_path = ''
    temperature = 0.5
    top_k = 12

    launch_gradio_server(server_name="0.0.0.0", server_port=7788)
