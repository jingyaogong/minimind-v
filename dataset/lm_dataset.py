import sys
import os
__package__ = "dataset"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import io
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from model.model_vlm import MiniMindVLM
import pyarrow.parquet as pq

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class VLMDataset(Dataset):
    def __init__(self, parquet_path, tokenizer, preprocess=None, max_length=512,
                 image_special_token='@' * 196):

        super().__init__()
        self.table = pq.read_table(parquet_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.image_token = image_special_token
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.table)

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
        conversations = json.loads(self.table['conversations'][index].as_py())
        image_bytes = self.table['image_bytes'][index].as_py()
        
        prompt = self._create_chat_prompt(conversations)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        loss_mask = self._generate_loss_mask(input_ids)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

        image = Image.open(io.BytesIO(image_bytes))
        image_tensor = MiniMindVLM.image2tensor(image, self.preprocess).unsqueeze(0)

        return X, Y, loss_mask, image_tensor

if __name__ == '__main__':
    import matplotlib.pyplot as plt; plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    for path in ['pretrain_data.parquet', 'sft_data.parquet']:
        t = pq.read_table(path); fig, ax = plt.subplots(1, 5, figsize=(20, 4))
        for i in range(5):
            ax[i].imshow(Image.open(io.BytesIO(t['image_bytes'][i].as_py()))); ax[i].axis('off')
            ax[i].set_title(json.loads(t['conversations'][i].as_py())[1]['content'][:30], fontsize=8)
        out = path.replace('.parquet', '_preview.png'); plt.savefig(out); print(f'已保存{out}, 共{len(t)}条')
