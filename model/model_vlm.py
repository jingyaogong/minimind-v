import os
import torch
import warnings
from .model_minimind import *
from typing import Optional, Tuple, List, Union
from torch import nn
from transformers import Siglip2ImageProcessor, Siglip2VisionModel
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

warnings.filterwarnings('ignore')


class VLMConfig(MiniMindConfig):
    model_type = "minimind-v"

    def __init__(self, image_special_token='<|image_pad|>', image_ids=[12], **kwargs):
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        self.image_hidden_size = kwargs.get("image_hidden_size", 768)
        self.image_token_len = kwargs.get("image_token_len", 64)
        super().__init__(**kwargs)

# 视觉投影层，把视觉特征投影到语言特征空间
class MMVisionProjector(nn.Module):
    def __init__(self, in_dim, out_dim, source_tokens=256, target_tokens=64):
        super().__init__()
        self.target_tokens = target_tokens
        #融合系数，把source_tokens个视觉token压缩为target个
        self.merge = source_tokens // target_tokens
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * self.merge, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        ) 
    def forward(self, x):
        b, n, d = x.shape
        #若输入的x.shape为(8, 256, 768)，则输出为(8, 256/4=64, 768*4=3072),压缩一张图片的视觉token数，最后还要通过mlp，把3072变成768，这其实有信息损失的问题，但是llm只需要上层的语义特征即可
        #torch的矩阵乘法只需要满足前一个矩阵最后一维和后一个矩阵的倒数第二维相同即可
        x = x.reshape(b, self.target_tokens, d * self.merge)
        return self.mlp(x)

# 继承自语言模型
class MiniMindVLM(MiniMindForCausalLM):
    #类属性，用于指定配置类
    config_class = VLMConfig

    def __init__(self, config: VLMConfig = None, vision_model_path="./model/siglip2-base-p16-ve"):
        #实例属性，用于存储配置
        self.config = config or VLMConfig()
        super().__init__(self.config)
        self.vision_encoder, self.processor = self.__class__.get_vision_model(vision_model_path)
        self.vision_proj = MMVisionProjector(self.config.image_hidden_size, self.config.hidden_size, target_tokens=self.config.image_token_len)

    #被init方法调用，self.processor接到get_vision_model的返回值，即processor
    @staticmethod #写给用户看的，说明这个函数是静态方法，不需要实例化类就可以调用
    def get_vision_model(model_path: str):
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        if not os.path.exists(model_path):
            return None, None
        model = Siglip2VisionModel.from_pretrained(model_path)
        processor = Siglip2ImageProcessor.from_pretrained(model_path)
        # 冻结 vision_encoder 的所有参数，现成的视觉模型不做训练
        for param in model.parameters():
            param.requires_grad = False
        #siglip跑的是推理模式
        return model.eval(), processor

    #图片通过processor转换为RGB格式的tensor
    @staticmethod
    def image2tensor(image, processor):
        #如果图片是RGBA或LA格式，则转换为RGB格式
        if image.mode in ['RGBA', 'LA']: image = image.convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        return inputs


    @staticmethod
    def get_image_embeddings(image_inputs, vision_model):
        #如果image_inputs有keys属性，说明是字典
        if hasattr(image_inputs, 'keys'):
            #等价于for k, v in image_inputs.items():
            #    if v.ndim > 2 and v.shape[1] == 1:
            #        v = v.squeeze(1)
            #    else:
            #        v = v
            #(1, 1, 3, 224, 224)→(1, 3, 224, 224)
            image_inputs = {k: v.squeeze(1) if v.ndim > 2 and v.shape[1] == 1 else v for k, v in image_inputs.items()}
        with torch.no_grad():
            outputs = vision_model(**image_inputs)
        #只要outputs大礼包里面的last_hidden_state（图像特征）
        return outputs.last_hidden_state

    @torch.compiler.disable
    #构造文本视觉融合输入
    #h和vf都是经过embedded后的特征，只不过一个是文本的，一个是视觉的
    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
        #self.config.image_ids是一个列表，通常只有1个元素，是视觉token的id索引
        #这两个条件说的是一个事情
        if vision_tensors is None or not self.config.image_ids:
            return h
        marker, vf = self.config.image_ids[0], vision_tensors
        #把vf从(8, 64, 768)变成(8, 1, 64, 768)
        if vf.dim() == 3:
            vf = vf.unsqueeze(1)
        out = []
        #遍历每个样本，h表示文本输入
        for b in range(h.size(0)):
            #h是text_tokens的特征(batch_size, seq_length, hidden_size)
            #tokens是inputs_ids(batch_size, seq_length)
            #Attention:VLM的
            hb, seq, k, i = h[b], tokens[b].tolist(), 0, 0
            while i < len(seq):
                if seq[i] == marker:
                    start = i
                    while i < len(seq) and seq[i] == marker:
                        i += 1
                    if k < vf.size(1):
                        #做一个拼接，中间放视觉特征，前后放文本特征，往往i-start就是target_tokens=64
                        #这里的k就是0，因为unsqueeze(1)，单文本只能对应单图
                        #不过可以拓展，只要不用前面unsqueeze(1)的逻辑，把第二个维度改成num_images，每次就代表着一个样本中的第几张图片
                        hb = torch.cat((hb[:start], vf[b][k][:i - start], hb[i:]), dim=0)[:seqlen]
                        #k是图片样本的索引
                        k += 1
                else:
                    #不是就继续找
                    i += 1
            out.append(hb)
        return torch.stack(out)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                labels: Optional[torch.Tensor] = None,
                pixel_values: Optional[torch.FloatTensor] = None,
                **args):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.model.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        #文本ids经过embedded得到hidden_states
        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        #图片经过embedded得到vision_tensors
        if pixel_values is not None and start_pos == 0:
            if hasattr(pixel_values, 'keys'):
                #这样写强调静态方法，不需要实例化类就可以调用。否则写为self.get_image_embeddings
                img_emb = MiniMindVLM.get_image_embeddings(pixel_values, self.vision_encoder)
                vision_tensors = self.vision_proj(img_emb)
            else:
                # 一段多图：N 表示一个样本中的图片数，[B, N, 1, C, H, W] -> [B, N, C, H, W]
                if len(pixel_values.shape) == 6:
                    pixel_values = pixel_values.squeeze(2)
                bs, num, c, im_h, im_w = pixel_values.shape

                # 每次取整个 batch 的第 i 张图，经过 vision encoder + projector 后得到 [B, target_tokens, hidden_size]
                # 再统一在图片维（dim=1）上堆叠，使 vision_tensors 始终为 [B, N, target_tokens, hidden_size]
                # 这样可与后续 count_vision_proj 中的 vf[b][k]、vf.size(1) 保持一致
                vision_tensors = torch.stack(
                    [self.vision_proj(MiniMindVLM.get_image_embeddings(pixel_values[:, i, :, :, :], self.vision_encoder))
                        for i in range(num)],dim=1)
            hidden_states = self.count_vision_proj(tokens=input_ids, h=hidden_states, vision_tensors=vision_tensors, seqlen=input_ids.shape[1])

        position_embeddings = (
            self.model.freqs_cos[start_pos:start_pos + seq_length],
            self.model.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.model.norm(hidden_states)

        aux_loss = sum([l.mlp.aux_loss for l in self.model.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        aux_loss = aux_loss + sum(p.sum() for p in self.vision_proj.parameters()) * 0  # dummy gradient for DDP
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        output = MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=presents, hidden_states=hidden_states)
        return output
