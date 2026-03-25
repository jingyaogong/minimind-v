<div align="center">

![logo](./images/logo.png)

</div>


<div align="center">

[![GitHub Repo stars](https://img.shields.io/github/stars/jingyaogong/minimind-v?style=social)](https://github.com/jingyaogong/minimind-v/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/jingyaogong/minimind-v?v=1)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/jingyaogong/minimind-v)](https://github.com/jingyaogong/minimind-v/commits/master)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/jingyaogong/minimind-v/pulls)
[![Collection](https://img.shields.io/badge/🤗-MiniMindV%20%20Collection-blue)](https://huggingface.co/collections/jingyaogong/minimind-v-67000833fb60b3a2e1f3597d)

</div>

<div align="center">

![GitHub Trend](https://trendshift.io/api/badge/repositories/13265)

</div>


<div align="center">
  <h3>"The Greatest Path is the Simplest"</h3>
</div>

<div align="center">

[中文](./README.md) | English

</div>

* This project aims to train a super-small multimodal vision-language model, **MiniMind-V**, with just a cost of 1.3 RMB
  and 1 hours of work, starting from scratch!
* The smallest version of **MiniMind-V** is only about $\frac{1}{2600}$ the size of GPT-3, designed to enable fast
  inference and even training on personal GPUs.
* **MiniMind-V** is an extension of the visual capabilities of the [MiniMind](https://github.com/jingyaogong/minimind)
  pure language model.
* The project includes full code for the minimalist structure of large VLM models, dataset cleaning, pretraining, and
  supervised fine-tuning (SFT).
* This is not only the smallest implementation of an open-source VLM model but also a concise tutorial for beginners in
  vision-language models.
* The hope is that this project can provide a useful example to inspire others and share the joy of creation, helping to
  drive progress in the wider AI community!

> To avoid misunderstandings, the "1 hours" is based on testing (`1 epoch`) with an NVIDIA 3090 hardware device (single GPU), and
> the "1.3 RMB" refers to GPU server rental costs. 

<div align="center">

![minimind2-v](./images/minimind-3v.gif)

[🔗🤖 Online Experience](https://www.modelscope.cn/studios/gongjy/MiniMind-V) | [🔗🎞️ Video Introduction](https://www.bilibili.com/video/BV1Sh1vYBEzY)

</div>

# 📌 Introduction

“Building a plane with Legos is much more exciting than flying in first class!”
Is it really as complex as imagined to build a VLM-based multimodal large model? How is the code implementation done?
Is the training process difficult? Now, let's explore the answers and feel the joy of creation together!

> [!TIP]
> (As of 2026-02-15) The MiniMind-V series has completed the training of the following model versions, with the smallest
> requiring only 67M (0.067B) parameters, capable of both image recognition and conversation!

| Model (Size)              | Inference Memory | Release    |
|---------------------------|------------------|------------|
| minimind-3v-moe (201M-A67M) | 1.0 GB           | 2026.04.01 |
| minimind-3v (67M)         | 0.5 GB           | 2026.04.01 |
| MiniMind2-V (104M)        | 1.1 GB           | 2025.02.20 |
| MiniMind2-Small-V (26M)   | 0.6 GB           | 2025.02.20 |
| minimind-v-v1-small (27M) | 0.6 GB           | 2024.10.04 |
| minimind-v-v1 (109M)      | 1.1 GB           | 2024.10.04 |

### 👉**Recent Updates**

<details> 
<summary> <b>2026-04-01</b> </summary>

- Added minimind-3v (67M) and minimind-3v-moe (201M-A67M) models
- Unified 768+8 architecture, supporting both dense and moe modes
- Switched Visual Encoder from CLIP to SigLIP2 (siglip2-base-p16-ve)
- Replaced QFormer with MLP Projection + reshape compression
- Dataset format updated to parquet, mixed data sources, updated tokenizer with image placeholder `<|image_pad|>`, new WebUI with dynamic model directory scanning and dropdown model switching
- Model code refactored, LLM/VLM unified for Transformers format
- Training scripts support DDP multi-GPU, bfloat16 mixed precision, torch.compile acceleration

</details>

<details> 
<summary> <b>2025-10-24</b> </summary>

- Bug fix: model weights mismatch
- Adapted to ["minimind-1024 update"](https://github.com/jingyaogong/minimind)
- Code refactoring: training and evaluation scripts standardized
- Added complete checkpoint resumption support

</details>

<details> 
<summary> <b>2025-04-27</b> </summary>

- Compatibility updates
- Adapted to the new feature in the "minimind" repository
- Standardized parts of the code

</details>

<details>

<summary> <b>More...</b> </summary>

**2025-02-20**

- MiniMind2-V updated alongside MiniMind2
- Significant reduction of all redundant code, standardized code format
- Major simplification of the model's redundant structure
- Updated dataset format, expanded with new SFT datasets
- Better performance than the previous VLM version!

**2024-10-05**

- MiniMind-V released on schedule, first open-source release

</details>

# 📌 Quick Start

<details>
<summary>Sharing my hardware and software configuration (for reference only)</summary>

* CPU: Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz
* RAM: 128 GB
* GPU: NVIDIA GeForce RTX 3090(24GB) * 8
* Ubuntu==20.04
* CUDA==12.2
* Python==3.10.16
* [requirements.txt](./requirements.txt)

</details>

### Step 0

```bash
# Clone the code repository
git clone --depth 1 https://github.com/jingyaogong/minimind-v
```

```bash
# Download the siglip2 model to the ./model directory
git clone https://huggingface.co/jingyaogong/siglip2-base-p16-ve
# or
git clone https://modelscope.cn/models/gongjy/siglip2-base-p16-ve
```

```bash
# Download the minimind language model to the ./out directory (as the base language model for training VLM):
# HuggingFace
https://huggingface.co/jingyaogong/minimind-3v-pytorch/blob/main/llm_768.pth
# Domestic source
https://modelscope.cn/models/gongjy/minimind-3v-pytorch/resolve/master/llm_768.pth
```


## Ⅰ Test an existing model's performance

### 1' Environment Preparation

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

### 2' Download the model

```bash
git clone https://huggingface.co/jingyaogong/minimind-3v
```

### 3' Command-line Q&A

```bash
# load_from='model': load native PyTorch weights, load_from='other path': load transformers format
python eval_vlm.py --load_from model --weight sft_vlm

# Or use transformers format model
python eval_vlm.py --load_from minimind-3v
```

### 4' Or start the WebUI

```bash
# ⚠️ You must first copy the transformers model folder to the ./scripts/ directory (e.g.: cp -r minimind-3v ./scripts/minimind-3v). The web_demo_vlm script will automatically scan subdirectories containing weight files; it will report an error if none are found.
cd scripts && python web_demo_vlm.py
```

## Ⅱ Train from scratch

### 1' Environment Preparation

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

<details>
<summary>Note: Test if Torch can use CUDA</summary>

```bash
import torch
print(torch.cuda.is_available())
```

If unavailable, download the whl file from [torch_stable](https://download.pytorch.org/whl/torch_stable.html) for
installation. Refer
to [this link](https://blog.csdn.net/weixin_45456738/article/details/141029610?ops_request_misc=&request_id=&biz_id=102&utm_term=%E5%AE%89%E8%A3%85torch&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-141029610.nonecase&spm=1018.2226.3001.4187)
for help.

</details>

### 2' Download Data

Download the required content from the [dataset link](https://huggingface.co/datasets/jingyaogong/minimind-v_dataset) 
and place it under `./dataset`.

<details>
<summary>Note: Dataset Details</summary>

**[Note 1]** Previously, extracting 500k fragmented image files could be very slow. From 2025-12-27, dataset format is unified to Parquet with image-text integrated storage, smaller size, no decompression needed, faster loading.

**[Note 2]** Parquet is a columnar storage format supporting efficient compression and fast reading. To preview data content, run `python lm_dataset.py` in the `dataset/` directory to visualize the first 5 image-text pairs.

Pretrain data:
```bash
wget https://hf-mirror.com/datasets/jingyaogong/minimind-v_dataset/resolve/main/pretrain_i2t.parquet
```

SFT data:
```bash
wget https://hf-mirror.com/datasets/jingyaogong/minimind-v_dataset/resolve/main/sft_i2t.parquet
```

Please reserve about ~2GB of space for the dataset. If there is insufficient space for pretrain data, you can try skipping the pretrain training step and proceed directly to SFT training.

</details>

### 3' Start Training

**3.1 Pretraining (Learning image description)**

```bash
# Basic training command (start from LLM weights, train vision_proj only)
python train_pretrain_vlm.py --epochs 4 --from_weight llm
```

> Run pretraining to get `pretrain_vlm_*.pth` as the pretrained model's output weights (* represents the model
> dimension, default is 768).

**3.2 Supervised Fine-Tuning (Learning image-caption dialogue style)**

```bash
# Basic training command (start from pretrain weights, full parameter fine-tuning)
python train_sft_vlm.py --epochs 2 --from_weight pretrain_vlm
```

> Perform supervised fine-tuning to get `sft_vlm_*.pth` as the output weights for the fine-tuned model.

<details>
<summary>Note: Training Details</summary>

**Training Features:**
- Support checkpoint resumption: add `--from_resume 1` parameter to continue from last interruption
- Support GPU count changes: automatically convert steps when GPU count changes during resumption
- Atomic saving: use temporary file + replacement mechanism to prevent weight corruption from interruption
- Each save generates `out/**.pth` (model weights) and `checkpoints/**_resume.pth` (training state) files

```bash
# To resume training after interruption, use the same command and add --from_resume 1
python train_sft_vlm.py --epochs 4 --from_resume 1
```

**Parameter Description:**
- `--from_weight`: base weight name (llm, pretrain_vlm, none, etc.)
- `--save_weight`: save weight prefix name
- `--from_resume`: whether to resume training (0=start from scratch, 1=continue from checkpoint)
- `--freeze_llm`: whether to freeze LLM parameters (pretrain use only)
- More details can be found in the code

</details>

---

### 4' Test the Model's Performance

Ensure that the model `*.pth` file you want to test is located in the `./out/` directory.
You can also directly download the pre-trained `*.pth` file
from [here](https://huggingface.co/jingyaogong/minimind-3v-pytorch).

```bash
# Test SFT model (default)
python eval_vlm.py --weight sft_vlm

# Test Pretrain model
python eval_vlm.py --weight pretrain_vlm
```

---

> [!TIP]
> The training scripts are based on PyTorch's native framework and support multi-card acceleration. If your device has
> N (N>1) GPUs:

Single-machine N-card training method (DDP, supports multi-machine multi-card cluster)

```bash
torchrun --nproc_per_node N train_xxx.py
```

<details>
<summary>Note: Other Details</summary>

Single-machine N-card training (DeepSpeed)

```bash
deepspeed --master_port 29500 --num_gpus=N train_xxx.py
```

You can enable wandb logging during training:

```bash
# You need to log in: wandb login
torchrun --nproc_per_node N train_xxx.py --use_wandb
# and
python train_xxx.py --use_wandb
```

By adding the `--use_wandb` parameter, you can log the training process, and after training is complete, you can view
the process on the wandb website. You can specify the project name and run name by modifying the `wandb_project`
and `wandb_run_name` parameters.

[Note]: After June 2025, the domestic network environment cannot directly connect to WandB. The MiniMind project by default switches to using [SwanLab](https://swanlab.cn/) as the training visualization tool (fully compatible with WandB API), that is, just change `import wandb` to `import swanlab as wandb`, no other changes are needed.

</details>

# 📌 VLM Detail

The base language model of MiniMind-V (VLM), MiniMind (LLM), comes from the twin
project [minimind](https://github.com/jingyaogong/minimind). For detailed information on the model structure, training
specifics, principles, and testing results, please refer to the [minimind](https://github.com/jingyaogong/minimind)
project. To reduce redundancy, the discussion on LLM-related topics is omitted here, assuming you have a basic
understanding of MiniMind (LLM).

> Even if you are not very familiar with the details of LLMs, you can still follow the "Quick Start" guide to train a
> MiniMind-V, as it remains unaffected and the repository focuses on the lowest cost for out-of-the-box use!

MiniMind-V's structure adds two submodules, a Visual Encoder and a feature projection, with a modality-mixing branch to
support inputs from multiple modalities:
![LLM-structure](./images/VLM-structure.jpg)
![LLM-structure](./images/VLM-structure-moe.jpg)


<details>
<summary> [Important] Some Interesting Thoughts </summary>

Let's take a moment to think about two questions:

* What is a **Large Language Model (LLM)**?
* What is a multimodal model?

[This article](https://www.jiqizhixin.com/articles/2024-09-15-3) perfectly aligns with my thoughts:  
Although the name "large language model" (LLM) contains the word "language," they are actually not closely related to
language; this is just a historical issue. A more accurate name would be self-regressive Transformer or something else.
LLMs are more of a general statistical modeling technology, mainly using a self-regressive Transformer to simulate token
flows. These tokens can represent text, images, audio, action choices, and even molecules—anything, really.  
Therefore, as long as the problem can be converted into a process of simulating a series of discrete tokens, LLM can
theoretically solve it. In fact, with the increasing maturity of large language model technologies, we may see more and
more problems falling under this modeling paradigm. In other words, the problem is fixed in using LLM to "predict the
next token," but the role and meaning of the tokens differ in each domain.

[ZJU-LiXi](https://person.zju.edu.cn/xilics#694283) has also mentioned a similar viewpoint (roughly stated below):  
Text, video, audio, actions, etc., are considered "multimodal" signals in human perception, but the term "modality" is
essentially just a classification concept based on how humans store information. Just like `.txt` and `.png` files,
though they differ in visual presentation and higher-level forms, they are fundamentally the same. The concept of "
multimodal" arose simply because humans need to categorize these signals based on different sensory dimensions.  
However, for machines, regardless of the signal's "modality," they are ultimately presented as a sequence of binary "
monomodal" numbers. Machines do not differentiate the origin of these signals; they just process and analyze the
information contained within these sequences.

Personally, I think **Generative Pretrained Transformer (GPT)** is a more fitting term than **Large Language Model (LLM)
**, and I prefer to use "GPT" to represent models in the LLM/VLM/GPT-like architecture series rather than to ride on
OpenAI's coattails.

To summarize what GPTs do in one sentence:

A GPT model predicts the next, next-next, next-next-next token, etc., based on the current token... until the model
outputs the end token; here, the "token" doesn’t necessarily have to be text!

```text
> For an LLM model, if we need to understand an "image," we just treat the "image" as a special "foreign language" that has never been encountered before, and translate it into the "LLM language" via a "foreign language dictionary."
> For an LLM model, if we need to understand "audio," we just treat "audio" as a special "foreign language" that has never been encountered before, and translate it into the "LLM language" via a "foreign language dictionary."
> ...
```

<u>**To obtain MiniMind-V, we only need to do these 2 things:**</u>

1. Use the **"foreign language dictionary"** that is good at translating images, to translate the image from the **"
   foreign language"** into a model-understandable **"LLM language."**
2. Fine-tune the LLM so that it and the **"foreign language dictionary"** go through a period of adaptation, thereby
   better understanding images.

The "foreign language dictionary" is referred to as the Visual Encoder model.  
Like LlaVA, Qwen-VL, and other visual language models, MiniMind-V now uses the open-source SigLIP2 series models as the
Visual Encoder.  
Specifically, we use [siglip2-base-p16-ve](https://huggingface.co/jingyaogong/siglip2-base-p16-ve), a Visual
Encoder based on the ViT-B/16 architecture for describing image-text information.  
The current SigLIP2 NaFlex vision encoder generates up to 256 patch tokens from the processor output as the input to the
encoder layer, which produces a 1×768 dimensional embedding vector for calculating error with the text.  
We don’t need the final embedding representation, so we only take the output from the encoder layer, which is the output
feature from the core ViT backbone.  
It receives 256×768 features from the previous layer, which are then reshaped by concatenating every 4 adjacent tokens into 1 (256×768 → 64×3072), then projected to the LLM's hidden dimension via a 2-layer MLP (Linear→GELU→Linear), resulting in 64 visual tokens input into MiniMind-V.
After obtaining the image encoder features, the integration with the LLM requires aligning the visual features to the LLM's text token dimension, and mapping the image features into the same space as text embeddings. In other
words, the image features and native visual tokens cannot be directly treated the same; they require cross-modal feature
alignment.

[LlaVA-1](https://arxiv.org/pdf/2304.08485) achieves good alignment with a simple linear transformation, [LlaVA-1.5](https://arxiv.org/pdf/2310.03744) upgrades to a 2-layer MLP. MiniMind-V adopts the same MLP Projection approach as LlaVA-1.5, combined with reshape for token compression.

![llava-structure](./images/llava-structure.png)

With that, the internal structural changes of MiniMind-V are now fully presented.

</details>


---

Next, let's briefly discuss the changes in the external input and output of MiniMind-V.

The input to the VLM is still a segment of text containing special `<image>` placeholders.  
After computing the text embedding, the vector generated by the image encoder can be projected onto the corresponding
embedding part of the placeholder, replacing the original placeholder embedding.  
For example:

```text
<image>\nWhat is in this image?
```

In `minimind-v`, the image is replaced by 64 `<|image_pad|>` tokens as placeholder (the 256 SigLIP2 patch features are compressed to 64 tokens via reshape+MLP),  
thus the `minimind-v` prompt becomes:

```text
<|image_pad|><|image_pad|>...<|image_pad|>(×64)\nWhat is this image describing?
```

After calculating the embedding and projection, the vision features replace the corresponding placeholder embeddings, and the rest of the computation is identical to the LLM part.

![input](./images/minimind-v-input.jpg)

At this point, all the details of `MiniMind-V` have been presented. The VLM model subclass inherits from `MiniMind` with only **minimal** changes, core algorithm modifications `< 50 lines`, very low migration difficulty. The specific implementation may differ from `LlaVA` and similar models, but the overall idea is consistent.

# 📌 Experiment

## Ⅰ Dataset

Original Source:
- [Chinese-LLaVA-Vision](https://huggingface.co/datasets/LinkSoul/Chinese-LLaVA-Vision-Instructions): Contains approximately 570,000 pre-trained images from CC-3M and COCO 2014
- [llava-en-zh-300k](https://huggingface.co/datasets/BUAADreamer/llava-en-zh-300k): Contains 300k instruction fine-tuning data and 150k images
- [LLaVA-SFT-665K](https://huggingface.co/datasets/csuhan/LLaVA-SFT-665K): Contains 665k instruction fine-tuning data

The dataset contains both Chinese and English data. The Q&A content has been translated, with better support for Chinese, further organized and resized (pretrain resolution 128×128, sft resolution 160×160).

(pretrain_i2t.parquet) Pre-training dataset format:

```text
Columns: conversations (json string), image_bytes (binary), image_names (string)

conversations example:
[
  {"role": "user", "content": "Provide a brief description of the given image.\n<image>"},
  {"role": "assistant", "content": "Olive oil is a healthy ingredient for free use."}
]
image_bytes: <binary image data>
```

(sft_i2t.parquet) Single image instruction fine-tuning dataset format:

```text
Columns: conversations (json string), image_bytes (binary), image_names (string)

conversations example:
[
  {"role": "user", "content": "What impact does the location of the alarm clock have on sleep quality?<image>"},
  {"role": "assistant", "content": "Place the digital alarm clock on the nightstand..."}
]
image_bytes: <binary image data>
```

> Note: sft_i2t.parquet contains ~580K samples in total, of which ~236K are image-text conversations (i2t) and ~346K are pure text conversations (t2t). The latter is used to preserve the model's base language capabilities.

Dataset download
link: ([ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind-v_dataset) | [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind-v_dataset))

## Ⅱ Training

Training is divided into two stages, both freezing the Visual Encoder gradients and only training the Projection and LLM parts.
Training is initialized from LLM pre-trained weights, with support for DDP multi-GPU training, mixed precision (bfloat16), torch.compile acceleration, and swanlab logging.

> train_pretrain_vlm

The pre-training stage learns general image knowledge from ~1.13M image-text description pairs (e.g., a deer is a deer, a dog is a dog).
This stage uses a higher learning rate (1e-4), max sequence length of 360, freezes the LLM main parameters, and only sets the Projection and LLM's layer 0 as learnable,
aiming to quickly establish a basic mapping from visual features to the language space while avoiding damage to the LLM's existing language capabilities.

> train_sft_vlm

The instruction fine-tuning stage learns real Q&A formats from ~580K samples, of which ~236K are image-text multi-turn conversations and ~346K are pure text conversations (to preserve LLM base capabilities).
This stage uses a lower learning rate (1e-5~1e-6), max sequence length of 768, unfreezes all Projection and LLM parameters for full fine-tuning,
enabling the model to conduct multi-turn conversations based on image content, while mitigating catastrophic forgetting through the mixed-in pure text data.

> Training Time and Loss Trend (for reference only)

Pretrain [768+8] (dense & moe)  
![input](./images/pretrain_loss.jpg)

SFT [768+8] (dense & moe)  
![input](./images/sft_loss.jpg)

## Ⅲ Model Weights

| Format | ModelScope | HuggingFace |
|---|---|---|
| Native PyTorch (`*.pth`) | [minimind-3v-pytorch](https://www.modelscope.cn/models/gongjy/minimind-3v-pytorch) | [minimind-3v-pytorch](https://huggingface.co/jingyaogong/minimind-3v-pytorch) |
| Transformers | [minimind-v collection](https://modelscope.cn/collections/MiniMind-V-42b841dde22d41) | [minimind-v collection](https://huggingface.co/collections/jingyaogong/minimind-v-67000833fb60b3a2e1f3597d) |

> Note: The Transformers version is the `MiniMind-V` model after single-image instruction fine-tuning

# 📌 Test

### Effect Test

#### Single Image Dialogue

<table>
  <thead>
    <tr>
      <th>image</th>
      <th>minimind-3v</th>
      <th>minimind-3v-moe</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <img src="./dataset/eval_images/airplane-flying-blue-sky.jpg" alt="airplane">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>The image features a white and black airplane parked on a grassy terrain. The layers of a building are likely to be filled with air traffic control, such as the building's tall building, the large building, or the overall pavement. There are also two airplanes parked in the background. The airplanes are displayed on a board, and the airplane is flying through the air while the black and white airplane is parked on the ground.</td>
      <td>The image features a large jetliner with a large airplane sitting on the ground. It is likely an airplane or an aircraft, possibly a flight jet or a runway.</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/birthday-cake-candles-table.jpg" alt="birthday-cake">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>The image features a white cake in an old-fashioned cake with cake placed on it. It is surrounded by a few cooked ingredients, including the wedding cake.</td>
      <td>The image features a white cake in an old-fashioned cake with cake placed on it. It is surrounded by a few cooked ingredients, including the wedding cake.</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/pizza-on-wooden-board.jpg" alt="pizza">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>The image depicts a delicious pizza pizza with fresh toppings, which are likely present in a slice of pizza. The pizza is perfectly crispy, as it has a crispy crust and slightly crispy, making it a delightful pizza presentation. The pizza is filled with fresh toppings, adding to the crispy crust. The pizza is also a bit scrambled, as it has a fresh topping, while the crispy crust is cooked with a pizza pan. The pizza is likely to be a pizza with its crunchy texture and flavorfully. The pizza is also a popular choice for pizza with others, and it is a filling and crispy crust.</td>
      <td>The image features a scenic burning pizza in a pasta-style board, surrounded by a wooden cabinet, a garner topping for cheese and vegetables. There is a small cabinet nearby, with a pan-familt, a pizza flatter topping. The pizza is situated on the left side of the board, with a bowl of olives placed on top of the side.</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/red-sports-car-road.jpg" alt="red-car">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>The image shows a yellow yellow and red turf coastline on a street, with a couple of cars and a yellow and red traffic lights in the background.</td>
      <td>The image features a couple of female vanity van lying on a car. The car is situated on the ground, surrounded by a wet furniture. The couple is seen in the middle of the car, possibly observing the scene.</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/row-of-colorful-houses.jpg" alt="colorful-houses">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>IoT tree is a holiday that is often associated with Christmas culture, history, and celebration. The tree has a unique black and white striped pattern, which features a sweet treat, a budget-friendly chocolate cake with brown spots. The cake is burnt and has a balcony with a sweet treat, with the rich, vibrant colors of the striped pattern. The tree has a rich and burnt color, while the rich and vibrant colors are visually appealing. The tree has a striped pattern, which adds to the overall atmosphere of the image.</td>
      <td>The image is a colorful scene of a colorful vintage house, with a large pink roof of the wall and a red color scheme. The colorful vintage house has a pink color, and the color scheme appears to be fine, with a red color scheme.</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/snow-mountain-lake-view.jpg" alt="snow-mountain">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>The image features a snowy mountain surrounded by a large, dense mountains. The large body of water is quite dense, with the body of water being hot and the water is sandy. The tall trees are swirls, and the tall trees are standing on the snow-covered ground. The body of water is also sink, creating a savanna-like appearance.</td>
      <td>The image is an image of a large mountain visible in a lake, surrounded by the idyllic mountains and the mountains. It appears to be a blanket in the ocean, with the river and idylis watching the sea. The mountains are lined with fresh sand and waves, adding a sense of tranquility to the scene.</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/street-food-hotpot-table.jpg" alt="street-food">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>The image shows a variety of cooking options, including baking meat, cupcakes, and spinach. The baked vegetables are rich in a variety of flavors, including grilled, sautéed, and baked vegetables. The presence of a baked vegetable with a variety of vegetables in different parts suggests a variety of options, including baking, cooking, and baking. The cooking process is highly recommended, with a mix of vegetables and baking times, making it an ideal choice for those who prefer a variety of cooking options. The cooking process is also highly compatible, with a variety of flavors and textures enjoying the cooking process.</td>
      <td>The image features a variety of freshwater salads, glasses, and coworkers displayed on a table. There is a mix of freshwater ingredients, likely a bun, which can be seen in the menu. The freshwater ingredients are placed on the table, and there is a portion of the coworkers displayed in the middle of the room. There is also a bowl filled with various ingredients.</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/three-kittens-basket.jpg" alt="kittens">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>The image features a table filled with people standing together, casing bars, and a pair of brown cats. The table is filled with pink and white cats.</td>
      <td>The image is a black and white detail of a miscellaneous bunch of toys, which is likely to be a part of a group or a similar artistic field.</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/tropical-beach-palm-tree.jpg" alt="tropical-beach">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>The image features a brown wooden coat.</td>
      <td>The image shows a sandy beach with an umbrella on top of a chair, providing a visual appeal for people to sit on.</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/yellow-school-bus-road.jpg" alt="school-bus">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>The image displays a group of people sitting on the bus. They are waiting to be cautious and attentive to their feet, which indicates they are likely to be cautious and followed by the bus.</td>
      <td>The image features a large collection of school buses, a brick-and-middle bus, and a stack of cars visible in the background. The school bus is situated next to a school bus, and there are several people watching the bus. The school bus is visible in the background, with one person standing behind the other, while the other person is watching the bus. The bus is positioned behind the school bus, creating a seamless and dynamic visual effect.</td>
    </tr>
  </tbody>
</table>

### Effect Summary:

Both models can identify image subjects (airplane, cake, car, beach, etc.), but commonly exhibit repetitive expressions and hallucinated details. Limited by model and data scale, the overall performance is at a stage of "understanding the gist but inaccurate on details".

Visual signals are treated as a special foreign language by LLMs, so the "language learning" ability highly depends on the LLM's capacity. The stronger the LLM, the more powerful the corresponding VLM, and the performance boost becomes significant.

#### Future Areas for Improvement:

```text
> Introduce dynamic resolution and Tile-based encoding (like LLaVA-NeXT) to break through the fixed resolution limit.
> Visual Encoder could be upgraded to stronger vision encoders for finer-grained image features.
> Extend multi-image understanding, video understanding, and Visual Grounding capabilities.
> ...
```

# 📌 Acknowledge

> [!TIP]
> If you find `MiniMind-V` helpful, please consider giving it a ⭐ on GitHub. <br/>
> Given the limited expertise, there may be unknown issues, and we welcome everyone to discuss, correct, or submit PRs
> to improve the project in Issues. <br/>
> Your support is the driving force behind continuous improvements to the project. Thank you!

## 🤝 [Contributors](https://github.com/jingyaogong/minimind-v/graphs/contributors)

<a href="https://github.com/jingyaogong/minimind-v/graphs/contributors">
  <img width="200" src="https://contrib.rocks/image?repo=jingyaogong/minimind-v" />
</a>

## 😊 Acknowledgments

<a href="https://github.com/xinyanghuang7"><b>@xinyanghuang7</b></a>: <a href="https://github.com/xinyanghuang7/minimind-v/tree/hxy">Multi-image VLM branch</a> | <a href="https://github.com/jingyaogong/minimind-v/tree/32cf4c5c01337231fd907b92d513de8945594263">Repository provided up to this version</a> 

<details> 
<summary> <b>Reference Links & Thanks to the following excellent papers or projects</b> </summary>

- No particular order
- [LlaVA](https://arxiv.org/pdf/2304.08485)
- [LlaVA-VL](https://arxiv.org/pdf/2310.03744)
- [Chinese-LLaVA-Vision-Instructions](https://huggingface.co/datasets/LinkSoul/Chinese-LLaVA-Vision-Instructions)

</details>

## 🫶Supporter

<a href="https://github.com/jingyaogong/minimind-v/stargazers">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://reporoster.com/stars/dark/jingyaogong/minimind-v"/>
      <source media="(prefers-color-scheme: light)" srcset="https://reporoster.com/stars/jingyaogong/minimind-v"/>
      <img alt="github contribution grid snake animation" src="https://reporoster.com/stars/jingyaogong/minimind-v"/>
    </picture>
</a>

<a href="https://github.com/jingyaogong/minimind-v/network/members">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://reporoster.com/forks/dark/jingyaogong/minimind-v"/>
      <source media="(prefers-color-scheme: light)" srcset="https://reporoster.com/forks/jingyaogong/minimind-v"/>
      <img alt="github contribution grid snake animation" src="https://reporoster.com/forks/jingyaogong/minimind-v"/>
    </picture>
</a>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=jingyaogong/minimind-v&type=Date&theme=dark"/>
  <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=jingyaogong/minimind-v&type=Date"/>
  <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=jingyaogong/minimind-v&type=Date"/>
</picture>

# 🎓 Citation

If you find MiniMind-V helpful in your research or work, please cite:

```bibtex
@misc{minimind-v,
  title = {MiniMind-V: Train a Tiny VLM from Scratch},
  author = {Jingyao Gong},
  year = {2024},
  url = {https://github.com/jingyaogong/minimind-v},
  note = {GitHub repository, accessed 2026}
}
```

# 📜 License

This repository is licensed under the [Apache-2.0 License](LICENSE).

