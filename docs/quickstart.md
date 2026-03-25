# Quick Start

This page will help you quickly get started with the MiniMind-V project.

## 📋 Requirements

- **Python**: 3.10+
- **PyTorch**: 1.12+
- **CUDA**: 12.2+ (optional, for GPU acceleration)
- **VRAM**: At least 8GB (24GB recommended)

!!! tip "Hardware Configuration Reference"
    - CPU: Intel i9-10980XE @ 3.00GHz
    - RAM: 128 GB
    - GPU: NVIDIA GeForce RTX 3090 (24GB)
    - Ubuntu: 20.04
    - CUDA: 12.2
    - Python: 3.10.16

## 🚀 Testing Existing Models

### Step 0: Preparation

```bash
# Clone the repository
git clone https://github.com/jingyaogong/minimind-v
cd minimind-v
```

```bash
# Download SigLIP2 model to ./model directory
git clone https://huggingface.co/jingyaogong/siglip2-base-p16-ve
# or
git clone https://modelscope.cn/models/gongjy/siglip2-base-p16-ve
```

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

!!! warning "Torch CUDA Check"
    After installation, test if Torch can use CUDA:
    ```python
    import torch
    print(torch.cuda.is_available())
    ```
    If unavailable, download the whl file from [torch_stable](https://download.pytorch.org/whl/torch_stable.html) for installation.

### Step 2: Download Model

Download pretrained models from HuggingFace or ModelScope:

```bash
# From HuggingFace
git clone https://huggingface.co/jingyaogong/minimind-3v

# Or from ModelScope
git clone https://www.modelscope.cn/models/gongjy/minimind-3v.git
```

### Step 3: Command Line Q&A

```bash
# load_from='model': load native PyTorch weights, load_from='other path': load transformers format
python eval_vlm.py --load_from model --weight sft_vlm

# Or use transformers format model
python eval_vlm.py --load_from minimind-3v
```

### Step 4: Start WebUI (Optional)

```bash
# ⚠️ You must first copy the transformers model folder to the ./scripts/ directory
# (e.g.: cp -r minimind-3v ./scripts/minimind-3v)
# The web_demo_vlm script will automatically scan subdirectories containing weight files
cd scripts && python web_demo_vlm.py
```

## 📝 Effect Testing

### Single Image Dialogue Examples

**Test Image 1: Pizza**

```text
Q: Describe the content of this image
A: The image depicts a delicious pizza with fresh toppings on a wooden board...
```

**Test Image 2: Snow Mountain**

```text
Q: Describe the content of this image
A: The image shows a beautiful snow-capped mountain reflected in a calm lake...
```

### Model Performance

| Model | Parameters | Inference Speed | Image Understanding |
|-------|-----------|-----------------|---------------------|
| minimind-3v-moe | 201M-A67M | Fast | 😊😊😊😊😊😊 |
| minimind-3v | 67M | Very Fast | 😊😊😊😊😊 |

## 🔧 Loading from PyTorch Weights

If you want to use native PyTorch model weights (`*.pth` files):

### Download Weight Files

Download the required weight files from:

- [HuggingFace - minimind-3v-pytorch](https://huggingface.co/jingyaogong/minimind-3v-pytorch)
- [ModelScope - minimind-3v-pytorch](https://www.modelscope.cn/models/gongjy/minimind-3v-pytorch)

Files needed:
- `sft_vlm_768.pth` or `sft_vlm_768_moe.pth` (SFT model weights)
- Optional: `pretrain_vlm_768.pth` or `pretrain_vlm_768_moe.pth` (pretrained model weights)

### Run Testing

```bash
# Test SFT model
python eval_vlm.py --weight sft_vlm

# Test pretrain model
python eval_vlm.py --weight pretrain_vlm
```

## 📊 Model Architecture

MiniMind-V adds Visual Encoder and Projection layers on top of the MiniMind language model:

![VLM-structure](images/VLM-structure.jpg)

### Core Components

1. **Visual Encoder (SigLIP2)**
   - Uses `siglip2-base-p16-ve` model
   - Based on SigLIP2 NaFlex processor
   - Output: up to 256×768 dimensional visual tokens

2. **Projection Layer**
   - Reshape (concat 4 adjacent tokens: 256×768 → 64×3072) + 2-layer MLP to 64 visual tokens
   - Aligns visual tokens to text embedding space

3. **Language Model (MiniMind)**
   - Inherits from MiniMind language model
   - Supports text generation and dialogue

### Model Parameter Configuration

| Model Name | Params | d_model | n_layers | kv_heads | q_heads | MoE |
|-----------|--------|---------|----------|----------|---------|-----|
| minimind-3v | 67M | 768 | 8 | 4 | 8 | No |
| minimind-3v-moe | 201M-A67M | 768 | 8 | 4 | 8 | Yes |

## 🎯 Next Steps

- Check [Model Training](training.md) to learn how to train your own vision-language model from scratch
- Read the source code to understand VLM implementation principles
- Try testing with your own images

## ❓ Common Issues

### 1. Model fails to load?

Ensure all dependency files are downloaded:
- SigLIP2 model weights
- MiniMind-V model weights
- tokenizer configuration files

### 2. Out of memory?

- Use the 67M dense version
- Reduce batch size
- Use CPU inference (slower)

### 3. Poor image recognition?

- Ensure image quality is clear
- Try adjusting image size
- Use more specific question descriptions

## 🔗 Related Resources

- **Online Demo**: [ModelScope Studio](https://www.modelscope.cn/studios/gongjy/MiniMind-V)
- **Video Introduction**: [Bilibili](https://www.bilibili.com/video/BV1Sh1vYBEzY)
- **Project Home**: [GitHub](https://github.com/jingyaogong/minimind-v)
