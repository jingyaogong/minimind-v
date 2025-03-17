import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from .model_vlm import MiniMindVLM
import os
import numpy as np
from torchvision import transforms

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

class VLMDataset(Dataset):
    def __init__(self, jsonl_path, images_path, tokenizer, preprocess=None, max_length=512,
                 image_special_token='@' * 196):

        super().__init__()
        self.samples = self.load_data(jsonl_path)
        self.images_path = images_path

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.image_token = image_special_token
        self.bos_id = tokenizer('<s>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('</s>\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content'].replace('<image>', self.image_token)})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_paths = sample['image']
        prompt = self._create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        loss_mask = self._generate_loss_mask(input_ids)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

        image_tensors = []
        for image_name in image_paths.split(','):
            image_name = image_name.strip()
            image = Image.open(f'{self.images_path}/{image_name}')
            image_tensor = MiniMindVLM.image2tensor(image, self.preprocess)
            image_tensors.append(image_tensor)
        image_tensors = torch.stack(image_tensors, dim=0)

        return X, Y, loss_mask, image_tensors

class T2IDataset(Dataset):
    def __init__(self, jsonl_path, images_path, tokenizer, max_length=512, img_pre_process=False,
                 image_special_token='@' * 256):

        super().__init__()
        self.samples = self.load_data(jsonl_path)
        self.images_path = images_path

        self.img_pre_process = img_pre_process # 是否使用提前处理的图片

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_size = 256
        self.image_token = image_special_token
        self.bos_id = tokenizer('<s>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('</s>\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content'].replace('<image>', self.image_token)})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_paths = sample['image']
        prompt = self._create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        loss_mask = self._generate_loss_mask(input_ids)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long) 

        if self.img_pre_process:
            image_emb_path = image_paths.replace('.jpg', '_emb.npy')
            image_emb = np.load(f'{self.images_path}/{image_emb_path}')
            image_emb = torch.tensor(image_emb, dtype=torch.float32)
            # 加载预处理的图像token
            image_token_path = image_paths.replace('.jpg', '_token.npy')
            image_token = np.load(f'{self.images_path}/{image_token_path}')
            image_token = torch.tensor(image_token, dtype=torch.long)
            return X, Y, loss_mask, (image_emb, image_token)
        else:
            image_tensors = []
            for image_name in image_paths.split(','):
                image_name = image_name.strip()
                image = Image.open(f'{self.images_path}/{image_name}').convert("RGB")
                image = center_crop_arr(image, self.image_size)
                image = np.array(image) / 255.
                image = 2.0 * image - 1.0
                image = torch.tensor(image, dtype=torch.float32)
                image_tensor = torch.einsum('hwc->chw', image)
                image_tensors.append(image_tensor)
            image_tensors = torch.stack(image_tensors, dim=0)   

            return X, Y, loss_mask, image_tensors
