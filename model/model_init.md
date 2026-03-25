## Download Vision Encoder (Required)

This model requires an additional vision encoder `siglip2-base-p16-ve`


```bash
git lfs install

# from HuggingFace
git clone https://huggingface.co/jingyaogong/siglip2-base-p16-ve

# or from ModelScope
git clone https://modelscope.cn/models/gongjy/siglip2-base-p16-ve
```

Directory Structure

```
model/
├── siglip2-base-p16-ve/            # Vision Encoder (Required)
│   ├── config.json
│   ├── model.safetensors           # ~180MB, float16
│   └── preprocessor_config.json
├── model_minimind.py               # LLM structure
├── model_vlm.py                    # VLM structure
├── tokenizer.json
└── tokenizer_config.json
```