import argparse
import torch
import warnings
import gradio as gr
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model_vlm import MiniMindVLM
from model.VLMConfig import VLMConfig
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()
warnings.filterwarnings('ignore')


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')

    if args.load == 0:
        moe_path = '_moe' if lm_config.use_moe else ''
        ckp = f'./out/sft_vlm_{lm_config.dim}{moe_path}.pth'
        model = MiniMindVLM(lm_config)
        state_dict = torch.load(ckp, map_location=args.device)
        model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=False)
    else:
        transformers_model_path = './MiniMind2-V'
        tokenizer = AutoTokenizer.from_pretrained(transformers_model_path)
        model = AutoModelForCausalLM.from_pretrained(transformers_model_path, trust_remote_code=True)

    print(f'VLM参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')

    vision_model, preprocess = MiniMindVLM.get_vision_model()
    return model.eval().to(args.device), tokenizer, vision_model.to(args.device), preprocess


def chat(prompt, current_image_path):
    global temperature, top_p
    image = Image.open(current_image_path).convert('RGB')
    pixel_tensors = MiniMindVLM.image2tensor(image, preprocess).to(model.device).unsqueeze(0)

    prompt = f'{lm_config.image_special_token}\n{prompt}'
    messages = [{"role": "user", "content": prompt}]

    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )[-args.max_seq_len + 1:]
    with torch.no_grad():
        x = torch.tensor(tokenizer(new_prompt)['input_ids'], device=model.device).unsqueeze(0)
        outputs = model.generate(
            x,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=args.max_seq_len,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            pad_token_id=tokenizer.pad_token_id,
            pixel_tensors=pixel_tensors
        )
        try:
            if not args.stream:
                print(tokenizer.decode(outputs.squeeze()[x.shape[1]:].tolist(), skip_special_tokens=True), end='')
            else:
                history_idx = 0
                for y in outputs:
                    answer = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
                    if (answer and answer[-1] == '�') or not answer:
                        continue
                    yield answer[history_idx:]
                    history_idx = len(answer)
        except StopIteration:
            print("No answer")
        print('\n')


def launch_gradio_server(server_name="0.0.0.0", server_port=7788):
    global temperature, top_p
    temperature = args.temperature
    top_p = args.top_p

    with gr.Blocks() as demo:
        gr.HTML(f"""
            <div style="text-align: center; margin-bottom: 1rem; display: flex; align-items: center; justify-content: center;">
                <img src="https://www.modelscope.cn/api/v1/studio/gongjy/MiniMind/repo?Revision=master&FilePath=images%2Flogo2.png&View=true" 
                     style="height: 60px;">
                <span style="margin: 0 0 0 1rem;font-size:40px;font-style: italic;font-weight:bold;">Hi, I'm MiniMind2-V</span>
            </div>
            """)

        with gr.Row():
            with gr.Column(scale=3):
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

                def update_parameters(temperature_, top_p_):
                    global temperature, top_p
                    temperature = float(temperature_)
                    top_p = float(top_p_)
                    return temperature, top_p

                with gr.Blocks() as iface_param:
                    with gr.Row():
                        temperature_slider = gr.Slider(label="Temperature", minimum=0.5, maximum=1.1, value=0.65)
                        top_p_slider = gr.Slider(label="Top-P", minimum=0.7, maximum=0.95, value=0.85)

                    temperature_slider.change(fn=update_parameters, inputs=[temperature_slider, top_p_slider])
                    top_p_slider.change(fn=update_parameters, inputs=[temperature_slider, top_p_slider])

            with gr.Column(scale=6):
                def chat_with_vlm(message, history):
                    if not message:
                        yield history + [("错误", "错误：提问不能为空。")]
                        return
                    if not current_image_path:
                        yield history + [("错误", "错误：图片不能为空。")]
                        return

                    image_html = f'<img src="gradio_api/file={current_image_path}" alt="Image" style="width:100px;height:auto;">'
                    res_generator = chat(message, current_image_path)
                    response = ''
                    for res in res_generator:
                        response += res
                        yield history + [(f"{image_html} {message}", response)]

                chatbot = gr.Chatbot(label="MiniMind-Vision", height=680)
                with gr.Row():
                    with gr.Column(scale=8):
                        message_input = gr.Textbox(
                            placeholder="请输入你的问题...",
                            show_label=False,
                            container=False
                        )
                    with gr.Column(scale=2, min_width=50):
                        submit_button = gr.Button("发送")
                submit_button.click(
                    fn=chat_with_vlm,
                    inputs=[message_input, chatbot],
                    outputs=chatbot
                )

                # # 添加示例问题
                # gr.Examples(
                #     examples=["描述一下这个图片的内容。", "画面里面有什么？", "画面里的天气怎么样？"],
                #     inputs=message_input)

        demo.launch(server_name=server_name, server_port=server_port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chat with MiniMind")
    parser.add_argument('--lora_name', default='None', type=str)
    parser.add_argument('--out_dir', default='out', type=str)
    parser.add_argument('--temperature', default=0.65, type=float)
    parser.add_argument('--top_p', default=0.85, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--dim', default=768, type=int)
    parser.add_argument('--n_layers', default=16, type=int)
    parser.add_argument('--max_seq_len', default=8192, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument('--stream', default=True, type=bool)
    parser.add_argument('--load', default=0, type=int, help="0: 原生torch权重，1: transformers加载")
    args = parser.parse_args()

    lm_config = VLMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)
    model, tokenizer, vision_model, preprocess = init_model(lm_config)
    launch_gradio_server(server_name="0.0.0.0", server_port=8888)
