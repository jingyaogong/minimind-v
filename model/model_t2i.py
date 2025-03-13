from .VLMConfig import VLMConfig
from .model import *
from typing import Optional, Tuple, List
from torch import nn
import warnings
from model.model_vlm import MiniMindVLM, VisionProj
from model.model_vq import VQ_models
import torch
from einops import rearrange

warnings.filterwarnings('ignore')


# 继承自语言模型
class MiniMindT2I(MiniMindVLM):
    config_class = VLMConfig

    def __init__(self, params: VLMConfig = None):
        super().__init__(params)
        if not params: params = VLMConfig()
        self.params = params
        self.image_tokenizer = self.__class__.get_image_tokenizer()
        # 这里重写是为了将output映射到codebook_size
        self.output = nn.Linear(params.dim, 16384, bias=False)
        self.vision_proj = VisionProj(ve_dim=8, lm_dim=params.dim)

    @staticmethod
    def get_image_tokenizer(model_path="./model/minimind_img_tokenizer/vq_ds16_t2i.pt"):
        model = VQ_models['VQ-16'](
            codebook_size=16384,
            codebook_embed_dim=8)
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location="cpu")
        if "ema" in checkpoint:  # ema
            model_weight = checkpoint["ema"]
        elif "model" in checkpoint:  # ddp
            model_weight = checkpoint["model"]
        elif "state_dict" in checkpoint:
            model_weight = checkpoint["state_dict"]
        else:
            raise Exception("please check model weight")
        model.load_state_dict(model_weight)
        del checkpoint

        # 冻结 image_tokenizer 的所有参数
        for param in model.parameters():
            param.requires_grad = False
        return model.eval()
    
    @staticmethod
    def get_image_embeddings(image_tensors, image_tokenizer):
        with torch.no_grad():
            # latent为离散化向量, indices为离散化向量的tokenid
            latent, _, [_, _, indices] = image_tokenizer.encode(image_tensors) 
        # 展平latent [B, C, H, W] -> [B, H*W, C]
        img_embedding = rearrange(latent, 'b c h w -> b (h w) c')
        indices = indices.reshape(-1, 256)
        return img_embedding, indices

    # 替换"@......"为图片的token
    def count_token_replace(self, tokens, vision_token=None, seqlen=512):
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

        if vision_token is not None and image_indices:
            new_tokens = []
            for i in range(tokens.size(0)):
                if i in image_indices:
                    token_i = tokens[i]
                    img_idx = 0
                    for start_idx, end_idx in image_indices[i]:
                        if img_idx < vision_token.size(1):
                            token_i = torch.cat(
                                (token_i[:start_idx], vision_token[i][img_idx], token_i[end_idx + 1:]), dim=0
                            )[:seqlen]
                            img_idx += 1
                    new_tokens.append(token_i)
                else:
                    new_tokens.append(tokens[i])

            return torch.stack(new_tokens, dim=0)
        return tokens

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                target_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **args):
        start_pos = args.get('start_pos', 0)
        pixel_tensors = args.get('pixel_tensors', None)
        h = self.tok_embeddings(input_ids)

        if pixel_tensors is not None and start_pos == 0:
            if len(pixel_tensors.shape) == 6:
                pixel_tensors = pixel_tensors.squeeze(2)
            bs, num, c, im_h, im_w = pixel_tensors.shape
            stack_dim = 1 if bs > 1 else 0
            # 获取图片的embedding
            vision_tensors = torch.stack([
                MiniMindT2I.get_image_embeddings(pixel_tensors[:, i, :, :, :], self.image_tokenizer)[0]
                for i in range(num)
            ], dim=stack_dim)
            h = self.count_vision_proj(tokens=input_ids, h=h, vision_tensors=vision_tensors, seqlen=input_ids.shape[1])
            # 获取图片的token
            vision_tokens = torch.stack([
                MiniMindT2I.get_image_embeddings(pixel_tensors[:, i, :, :, :], self.image_tokenizer)[1]
                for i in range(num)
            ], dim=stack_dim)
            # 替换target中的'@......'为图片的token
            target_ids = self.count_token_replace(tokens=target_ids, vision_token=vision_tokens, seqlen=target_ids.shape[1])

        pos_cis = self.pos_cis[start_pos:start_pos + input_ids.shape[1]]
        past_kvs = []
        for l, layer in enumerate(self.layers):
            h, past_kv = layer(
                h, pos_cis,
                past_key_value=past_key_values[l] if past_key_values else None,
                use_cache=use_cache
            )
            past_kvs.append(past_kv)

        logits = self.output(self.norm(h))
        aux_loss = sum(l.feed_forward.aux_loss for l in self.layers if isinstance(l.feed_forward, MOEFeedForward))

        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        self.OUT.__setitem__('target_ids', target_ids) # 添加target_ids
        return self.OUT
