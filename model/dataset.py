import json
import random
import re

import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from model.vision_utils import get_img_process
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    def __init__(self, json_path, tokenizer, vision_model=None, max_length=1024,
                 prompt_max_len=512,
                 answer_max_len=256,
                 image_special_token='<' * 25 + '>' * 25):

        super().__init__()
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.max_length = max_length
        self.prompt_max_len = prompt_max_len
        self.answer_max_len = answer_max_len
        self.tokenizer = tokenizer
        self.vision_model, self.preprocess = vision_model
        self.padding = 0
        self.bos_id = self.tokenizer('<s>assistant').data['input_ids']
        self.dataset_path = './dataset/pretrain_images/'
        self.image_special_token = image_special_token

    def __len__(self):
        return len(self.data)

    def find_sublist_index(self, main_list, sub_list) -> int:
        last_index = -1
        for i in range(len(main_list) - len(sub_list) + 1):
            if main_list[i:i + len(sub_list)] == sub_list:
                last_index = i
        return last_index

    def safe_eval(self, s):
        try:
            res = eval(s)
        except Exception as e:
            return []
        return res

    def __getitem__(self, index: int):
        sample = self.data[index]
        image_name = sample['image']
        conversation = sample['conversations']
        # minimind-v的image的特殊占位符，对应每张图切分成10个token，和get_img_process中的数量对应
        messages = []
        # 遍历 conversation 列表
        for i in range(0, len(conversation), 2):
            # 检查是否有配对的问题和回答
            if i + 1 < len(conversation):
                q = conversation[i]['value'].replace('<image>', self.image_special_token)
                a = conversation[i + 1]['value']

                if q and a:
                    messages.append({"role": "user", "content": q})
                    messages.append({"role": "assistant", "content": a})

        new_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        input_id = self.tokenizer(new_prompt).data['input_ids'][:self.max_length]

        question_length = self.find_sublist_index(input_id, self.bos_id) + len(self.bos_id)
        padding_len = self.max_length - len(input_id)
        input_id = input_id + [self.padding] * padding_len
        mask_len = len(input_id) - question_length - padding_len
        loss_mask = [0] * question_length + [1] * (mask_len) + [0] * padding_len

        input_id = np.array(input_id)

        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)

        X_tensor = torch.from_numpy(X)
        Y_tensor = torch.from_numpy(Y)
        loss_mask_tensor = torch.from_numpy(loss_mask)

        image = Image.open(f'{self.dataset_path}{image_name}')
        image_encoders = get_img_process(image, self.preprocess)

        return X_tensor, Y_tensor, loss_mask_tensor, image_encoders


class SFTDataset(Dataset):
    def __init__(self, json_path, tokenizer, vision_model=None, max_length=1024,
                 prompt_max_len=512,
                 answer_max_len=256,
                 image_special_token='<' * 25 + '>' * 25):

        super().__init__()
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.max_length = max_length
        self.prompt_max_len = prompt_max_len
        self.answer_max_len = answer_max_len
        self.tokenizer = tokenizer
        self.vision_model, self.preprocess = vision_model
        self.padding = 0
        self.bos_id = self.tokenizer('<s>assistant').data['input_ids']
        self.dataset_path = './dataset/sft_images/'
        self.image_special_token = image_special_token

    def __len__(self):
        return len(self.data)

    def find_sublist_index(self, main_list, sub_list) -> int:
        last_index = -1
        for i in range(len(main_list) - len(sub_list) + 1):
            if main_list[i:i + len(sub_list)] == sub_list:
                last_index = i
        return last_index

    def safe_eval(self, s):
        try:
            res = eval(s)
        except Exception as e:
            return []
        return res

    def __getitem__(self, index: int):
        sample = self.data[index]
        image_name = 'COCO_train2014_' + sample['image']
        conversation = sample['conversations']
        # minimind-v的image的特殊占位符，对应每张图切分成M个token，和get_img_process中的数量对应
        messages = []
        # 遍历 conversation 列表
        # for i in range(0, len(conversation), 2):
        for i in range(0, 1):
            # 检查是否有配对的问题和回答
            if i + 1 < len(conversation):
                q = conversation[i]['value'].replace('<image>', self.image_special_token)
                a = conversation[i + 1]['value']

                if q and a:
                    messages.append({"role": "user", "content": q})
                    messages.append({"role": "assistant", "content": a})

        new_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        input_id = self.tokenizer(new_prompt).data['input_ids'][:self.max_length]

        question_length = self.find_sublist_index(input_id, self.bos_id) + len(self.bos_id)
        padding_len = self.max_length - len(input_id)
        input_id = input_id + [self.padding] * padding_len
        mask_len = len(input_id) - question_length - padding_len
        loss_mask = [0] * question_length + [1] * (mask_len) + [0] * padding_len

        input_id = np.array(input_id)

        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)

        X_tensor = torch.from_numpy(X)
        Y_tensor = torch.from_numpy(Y)
        loss_mask_tensor = torch.from_numpy(loss_mask)

        image = Image.open(f'{self.dataset_path}{image_name}')
        image_encoders = get_img_process(image, self.preprocess)

        return X_tensor, Y_tensor, loss_mask_tensor, image_encoders


if __name__ == "__main__":
    pass
