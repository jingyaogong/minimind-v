import os

import torch
import warnings

from transformers.generation.utils import GenerateOutput

from .model_minimind import *
from typing import Optional, Tuple, List, Callable
from torch import nn
from transformers import CLIPProcessor, CLIPModel, GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from typing import List

warnings.filterwarnings('ignore')


class VLMConfig(MiniMindConfig):
    model_type = "minimind-v"

    def __init__(
            self,
            image_special_token: str = '@' * 196,
            image_ids: List = [34] * 196,
            **kwargs,
    ):
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        super().__init__(**kwargs)


class VisionProj(nn.Module):
    def __init__(self, ve_hidden_size=768, hidden_size=512):
        super().__init__()
        self.ve_hidden_size = ve_hidden_size
        self.hidden_size = hidden_size
        self.vision_proj = nn.Sequential(
            nn.Linear(self.ve_hidden_size, self.hidden_size)
        )

    def forward(self, image_encoders):
        vision_proj = self.vision_proj(image_encoders)
        return vision_proj


# 继承自语言模型
class MiniMindVLM(MiniMindForCausalLM):
    config_class = VLMConfig

    def __init__(self, params: VLMConfig = None, vision_model_path="./model/vision_model/clip-vit-base-patch16"):
        super().__init__(params)
        if not params: params = VLMConfig()
        self.params = params
        self.vision_encoder, self.processor = self.__class__.get_vision_model(vision_model_path)
        self.vision_proj = VisionProj(hidden_size=params.hidden_size)

    @staticmethod
    def get_vision_model(model_path: str):
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        if not os.path.exists(model_path):
            return None, None
        model = CLIPModel.from_pretrained(model_path)
        processor = CLIPProcessor.from_pretrained(model_path)
        # 冻结 vision_encoder 的所有参数
        for param in model.parameters():
            param.requires_grad = False
        return model.eval(), processor

    @staticmethod
    def image2tensor(image, processor):
        if image.mode in ['RGBA', 'LA']: image = image.convert('RGB')
        inputs = processor(images=image, return_tensors="pt")['pixel_values']
        return inputs

    @staticmethod
    def get_image_embeddings(image_tensors, vision_model):
        with torch.no_grad():
            outputs = vision_model.vision_model(pixel_values=image_tensors)
        img_embedding = outputs.last_hidden_state[:, 1:, :].squeeze()
        return img_embedding

    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
        def find_indices(tokens, image_ids):
            image_ids_tensor = torch.tensor(image_ids).to(tokens.device)
            len_image_ids = len(image_ids)
            if len_image_ids > tokens.size(1):
                return None
            tokens_view = tokens.unfold(1, len_image_ids, 1)
            matches = (tokens_view == image_ids_tensor).all(dim=2)
            return {
                batch_idx: [(idx.item(), idx.item() + len_image_ids - 1) for idx in
                            matches[batch_idx].nonzero(as_tuple=True)[0]]
                for batch_idx in range(tokens.size(0)) if matches[batch_idx].any()
            } or None

        image_indices = find_indices(tokens, self.params.image_ids)
        if vision_tensors is not None and image_indices:
            vision_proj = self.vision_proj(vision_tensors)
            if len(vision_proj.shape) == 3:
                vision_proj = vision_proj.unsqueeze(0)
            new_h = []
            for i in range(h.size(0)):
                if i in image_indices:
                    h_i = h[i]
                    img_idx = 0
                    for start_idx, end_idx in image_indices[i]:
                        if img_idx < vision_proj.size(1):
                            h_i = torch.cat((h_i[:start_idx], vision_proj[i][img_idx], h_i[end_idx + 1:]), dim=0)[
                                  :seqlen]
                            img_idx += 1
                    new_h.append(h_i)
                else:
                    new_h.append(h[i])
            return torch.stack(new_h, dim=0)
        return h

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                pixel_values: Optional[torch.FloatTensor] = None,
                **args):
        batch_size, seq_length = input_ids.shape
        past_key_values = past_key_values or [None] * len(self.model.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        if pixel_values is not None and start_pos == 0:
            if len(pixel_values.shape) == 6:
                pixel_values = pixel_values.squeeze(2)
            bs, num, c, im_h, im_w = pixel_values.shape
            stack_dim = 1 if bs > 1 else 0
            vision_tensors = torch.stack([
                MiniMindVLM.get_image_embeddings(pixel_values[:, i, :, :, :], self.vision_encoder)
                for i in range(num)
            ], dim=stack_dim)
            hidden_states = self.count_vision_proj(tokens=input_ids, h=hidden_states, vision_tensors=vision_tensors,
                                                   seqlen=input_ids.shape[1])

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

        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.model.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        self.OUT.__setitem__('last_hidden_state', hidden_states)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', presents)
        return self.OUT


    ## 以下函数使用gpt生成
    def sample_logits(self, logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0) -> torch.Tensor:
        if temperature != 1.0:
            logits = logits / (temperature + 1e-6)

        probs = F.softmax(logits, dim=-1)

        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False

            indices_to_remove = torch.zeros_like(probs, dtype=torch.bool)
            indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)

            probs = probs.masked_fill(indices_to_remove, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        return torch.multinomial(probs, num_samples=1)

    @torch.inference_mode()
    def generate(
            self,
            input_ids: torch.LongTensor,
            pixel_values: Optional[torch.FloatTensor] = None,
            max_new_tokens: Optional[int] = 128,
            do_sample: bool = False,
            temperature: float = 1.0,
            top_p: float = 1.0,
            num_return_sequences: int = 1,
            attention_mask: Optional[torch.Tensor] = None,
            pad_token_id: int = 0,
            eos_token_id: int = 2,
            streamer=None,
            use_cache: bool = True,
            **kwargs
    ) -> Union[GenerateOutput, torch.LongTensor]:



        generated = []

        for batch_idx in range(input_ids.size(0)):
            input_ids_i = input_ids[batch_idx][input_ids[batch_idx] != pad_token_id].unsqueeze(0)
            pixel_values_i = pixel_values[batch_idx:batch_idx + 1] if pixel_values is not None else None
            for _ in range(num_return_sequences):
                output = self._stream(
                    input_ids=input_ids_i,
                    pixel_values=pixel_values_i,
                    eos_token_id=eos_token_id,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    use_cache=use_cache,
                    do_sample=do_sample,
                    streamer=streamer,
                )
                tokens = []
                for t in output:
                    if streamer is None:
                        tokens.append(t[:, -1:])
                if tokens:
                    result = torch.cat(tokens, dim=1)
                    full_sequence = torch.cat([input_ids_i, result], dim=1)
                else:
                    full_sequence = input_ids_i
                generated.append(full_sequence)

        max_len = max(seq.shape[1] for seq in generated)
        generated = [
            torch.cat([
                seq,
                torch.full((1, max_len - seq.shape[1]), pad_token_id, dtype=seq.dtype, device=seq.device)
            ], dim=1)
            for seq in generated
        ]
        return torch.cat(generated, dim=0)

    def _stream(
            self,
            input_ids: torch.LongTensor,
            pixel_values: Optional[torch.FloatTensor] = None,
            eos_token_id: Optional[int] = 2,
            max_new_tokens: int = 128,
            temperature: float = 1.0,
            top_p: float = 1.0,
            do_sample: bool = False,
            streamer=None,
            use_cache: bool = True,
            **kwargs
    ):
        past_key_values = None
        first_pass = True
        start_pos = input_ids.shape[1]
        for _ in range(max_new_tokens):
            if first_pass:
                out = self(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    past_key_values=None,
                    use_cache=use_cache,
                    **kwargs
                )
                first_pass = False
            else:
                out = self(
                    input_ids=input_ids[:, -1:],  # only last token
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    **kwargs
                )

            logits = out.logits[:, -1, :]
            past_key_values = out.past_key_values

            if do_sample:
                next_token = self.sample_logits(logits)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if streamer is not None:
                streamer.put(next_token)

            yield input_ids[:, start_pos:]

            if next_token.item() == eos_token_id:
                break
