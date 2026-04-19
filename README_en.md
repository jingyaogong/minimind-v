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

* This project aims to train a super-small multimodal vision-language model, **MiniMind-V**, with just a cost of 3 RMB
  and 2 hours of work, starting from scratch!
* The smallest version of **MiniMind-V** is only about $\frac{1}{2600}$ the size of GPT-3, designed to enable fast
  inference and even training on personal GPUs.
* **MiniMind-V** is an extension of the visual capabilities of the [MiniMind](https://github.com/jingyaogong/minimind)
  pure language model.
* The project includes full code for the minimalist structure of large VLM models, dataset cleaning, Pretrain, and SFT.
* This is not only the smallest implementation of an open-source VLM model but also a concise tutorial for beginners in
  vision-language models.
* The hope is that this project can provide a useful example to inspire others and share the joy of creation, helping to
  drive progress in the wider AI community!

> Note: this project is released under the Apache 2.0 license and is completely free. The "2 hours" refer to the measured time of running `1 epoch` of SFT on a single NVIDIA 3090, and the "3 RMB" refer to the GPU rental cost for that time slot.

<div align="center">

![minimind-3v](./images/minimind-3v.gif)

[🔗🤖 Online Experience](https://www.modelscope.cn/studios/gongjy/MiniMind-V) | [🔗🎞️ Video Introduction](https://www.bilibili.com/video/BV1Sh1vYBEzY)

</div>

# 📌 Introduction

“Building a plane with Legos is much more exciting than flying in first class!”
Is it really as complex as imagined to build a VLM-based multimodal large model? How is the code implementation done?
Is the training process difficult? Now, let's explore the answers and feel the joy of creation together!

> [!TIP]
> (As of 2026-04-20) The MiniMind-V series has completed the training of the following model versions, with the smallest
> requiring only 67M (0.067B) parameters, capable of both image recognition and conversation!

| Model (Size) | Release |
|---|---|
| minimind-3v-moe (201M-A67M) | 2026.04.20 |
| minimind-3v (67M) | 2026.04.20 |
| MiniMind2-V (104M) | 2025.02.20 |
| MiniMind2-Small-V (26M) | 2025.02.20 |
| minimind-v-v1-small (27M) | 2024.10.04 |
| minimind-v-v1 (109M) | 2024.10.04 |

### 👉**Recent Updates**

<details>
<summary> <b>2026-04-20</b> </summary>

- New checkpoints released: minimind-3v (67M) / minimind-3v-moe (201M-A67M)
- Projector: added `LayerNorm`, token merging switched to 2D pixel-shuffle
- Vision Encoder switched to `SiglipVisionModel` (fixed 256×256)
- Training data moved to ALLaVA-4V (Pretrain 1.27M / SFT 2.9M, merged into a single-stage SFT)
- Freeze strategy updated: `freeze_llm=1` unfreezes first + last layers; Pretrain/SFT defaults now `2`/`1`; `max_seq_len` 360 → 450
- Misc bugfixes and small tweaks

</details>

<details> 
<summary> <b>2026-04-01</b> </summary>

- Added minimind-3v (67M) and minimind-3v-moe (201M-A67M) models
- Unified 768+8 architecture, supporting both dense and moe modes
- Switched Visual Encoder from CLIP to SigLIP2 (siglip2-base-p16-256-ve)
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

<summary> <b>More...</b> </summary>

**2025-04-27**

- Compatibility updates
- Adapted to [MiniMind repository new features](https://github.com/jingyaogong/minimind/issues/370)
- Code normalization

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
git clone https://huggingface.co/jingyaogong/siglip2-base-p16-256-ve
# or
git clone https://modelscope.cn/models/gongjy/siglip2-base-p16-256-ve
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

Pretrain data (optional; contains caption subset only):
```bash
wget https://hf-mirror.com/datasets/jingyaogong/minimind-v_dataset/resolve/main/pretrain_i2t.parquet
```

SFT data (required; contains full caption + instruct + text merge):
```bash
wget https://hf-mirror.com/datasets/jingyaogong/minimind-v_dataset/resolve/main/sft_i2t.parquet
```

The single `sft_i2t.parquet` file (2.9M rows) absorbs Pretrain as a subset, and after global dictionary encoding dedup it is only ~10% larger than the original SFT — enough to cover every training stage.

</details>

### 3' Start Training

**3.1 Pretrain (optional)**

> `sft_i2t.parquet` already contains the Pretrain data as a subset, so **this stage can be skipped** and you may go directly to SFT with `--from_weight llm`. If you prefer the Projector to be pre-aligned first and SFT to converge more stably, run a round of Pretrain separately.

```bash
# Basic training command (start from LLM weights, train vision_proj only)
python train_pretrain_vlm.py --epochs 2 --from_weight llm
```

> Run Pretrain to get `pretrain_vlm_*.pth` as the Pretrain output weights (* is the model dimension, default 768).

**3.2 SFT (required)**

```bash
# Basic command (default --freeze_llm 1: unfreeze proj + first/last LLM layers, keep middle layers' original LLM weights)
# If 3.1 was run: --from_weight pretrain_vlm; if Pretrain was skipped: --from_weight llm
python train_sft_vlm.py --epochs 2 --from_weight pretrain_vlm
```

> Run SFT to get `sft_vlm_*.pth` as the SFT output weights.

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
- `--freeze_llm`: freezing strategy (0=all trainable, 1=proj + first/last LLM layers, 2=proj only). Default 2 for Pretrain, 1 for SFT
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

The language backbone of MiniMind-V is the `llm_768.pth` trained by the sibling project [minimind](https://github.com/jingyaogong/minimind). The LLM's own structure, training details and experimental analysis are not repeated here; the default assumption is that the reader has a basic understanding of MiniMind LLM. Not having touched the LLM project does not prevent following the "Quick Start" to get MiniMind-V running — the flow is self-contained.

The two shorthand labels on the landing page — "from scratch" and "67M" — also need a stricter reading here. "From scratch" means the VLM itself is trained from zero (Projection randomly initialized, first/last LLM layers fine-tuned for alignment), but the LLM backbone is not pretrained from zero — it is continued from the weights of MiniMind. For a strictly "from-zero pretraining" setup, first pretrain an LLM in MiniMind and then plug it back here. "67M" refers to the trainable backbone (LLM ~64M + Projection ~3M); the SigLIP2 vision encoder contributes another ~93M parameters that stay frozen throughout and serve only as an image feature extractor, so the full model at inference time is roughly 160M (dense) / 294M (MoE).

The VLM adds a Visual Encoder and a feature projection on top of the LLM, introducing a modality-mixing branch to support multimodal inputs:
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
Specifically, we use [siglip2-base-p16-256-ve](https://huggingface.co/jingyaogong/siglip2-base-p16-256-ve), a Visual
Encoder based on the ViT-B/16 architecture for describing image-text information.  
The current SigLIP2 NaFlex vision encoder generates up to 256 patch tokens from the processor output as the input to the
encoder layer, which produces a 1×768 dimensional embedding vector for calculating error with the text.  
We don’t need the final embedding representation, so we only take the output from the encoder layer, which is the output
feature from the core ViT backbone.  
It receives 256×768 features from the previous layer, which are then reshaped by concatenating every 4 adjacent tokens into 1 (256×768 → 64×3072), then projected to the LLM's hidden dimension via a 2-layer MLP (Linear→GELU→Linear), resulting in 64 visual tokens fed into MiniMind-V — this step is exactly cross-modal feature alignment: the native visual features are brought into the semantic space where text tokens live, so that the two can interact in the same space.

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

All image-text data used in this round come from the [ALLaVA-4V](https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V) family.
Compared with data stitched together from earlier LLaVA-derived sets, ALLaVA-4V is more consistent in quality, natively paired in Chinese and English, and more thorough in its fine-grained descriptions.
It is composed of two image sources: a curated subset of LAION (mostly natural images) and a curated subset of VFLAN (documents, charts, synthetic scenes).

- **Pretrain (`pretrain_i2t.parquet`, ~1.27M rows / ~640K unique images)**
  - `ALLaVA-Caption-LAION-4V` en/zh: ~470K + ~440K
  - `ALLaVA-Caption-VFLAN-4V` en/zh: ~195K + ~170K
  - Single-turn "describe this image" style captions used to establish the basic alignment from visual tokens to language tokens.

- **SFT (`sft_i2t.parquet`, ~2.90M rows / ~650K unique images)**
  - `ALLaVA-Instruct-LAION-4V` en/zh: ~470K + ~470K
  - `ALLaVA-Instruct-VFLAN-4V` en/zh: ~195K + ~165K
  - `Instruct-LAION-4v-gemini-claude-ensembled` (synthesized by Gemini/Claude): ~50K
  - `Instruct-LAION-4oiterative` (iteratively refined by GPT-4o): ~50K
  - Pure-text conversations (8×8 black placeholder images, preserving base language ability): ~230K
  - **Full Pretrain caption data merged in** (same source as pretrain, ~99% image overlap): ~1.27M
  - A blend of "image-grounded reasoning Q&A", "caption-style long descriptions" and "pure-text chat" — covering fine-grained follow-ups/long chains of thought as well as image description and general language ability.

Roughly 2.9M samples in total. The Pretrain stage can be skipped entirely (SFT has absorbed it as a subset). Chinese and English are roughly balanced.
Given MiniMind-V's trainable backbone is only 67M, mixing English and Chinese is a pragmatic choice: Chinese data helps native-language generation, while the original English descriptions tend to be more precise — the two complement each other.

All images are `resize`d to **256×256** (matching SigLIP2 NaFlex's 256 patch-token input) and re-encoded as JPEG, packed directly into parquet.

(`pretrain_i2t.parquet`) Pre-training dataset format:

```text
Columns: conversations (json string), image_bytes (binary)

conversations example:
[
  {"role": "user", "content": "<image>\nPlease describe this image in detail."},
  {"role": "assistant", "content": "The image shows..."}
]
image_bytes: <binary image data>
```

(`sft_i2t.parquet`) Single-image SFT dataset format:

```text
Columns: conversations (json string), image_bytes (binary)

conversations example:
[
  {"role": "user", "content": "Based on the image, what time of day is it?<image>"},
  {"role": "assistant", "content": "Judging from the light and shadows..."}
]
image_bytes: <binary image data>
```

> Note: sft_i2t.parquet contains ~2.9M samples, of which ~1.40M are image instruct conversations, ~1.27M are image caption descriptions (merged from Pretrain), and ~230K are pure-text conversations (t2t, image column filled by an 8×8 black placeholder) used to preserve the model's base language capabilities. Since Pretrain is already included as a subset, the Pretrain stage can be skipped and SFT run directly.

Dataset download link: ([ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind-v_dataset) | [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind-v_dataset))

## Ⅱ Training

Training has two stages (Pretrain optional; SFT required). Both freeze the Visual Encoder and train only the Projection and part of the LLM layers.
Training is initialized from LLM Pretrain weights, with support for DDP multi-GPU training, mixed precision (bfloat16), torch.compile acceleration, and swanlab logging.

> train_pretrain_vlm (optional)

The Pretrain stage learns general image knowledge from ~1.27M image-text description pairs (e.g., a deer is a deer, a dog is a dog).
It uses a higher learning rate (~4e-4), max sequence length 450, and **fully freezes the LLM and Visual Encoder, training only the Projection** (`--freeze_llm 2`).
The goal is to let the Projector cleanly align visual tokens to the language space without perturbing the original LLM weights.
Since SFT data already contains all Pretrain samples as a subset, this stage is optional; skipping it saves time, while **running one round of Pretrain first lets the Projector pre-align and makes SFT converge more steadily**.

> train_sft_vlm

The SFT stage uses the aforementioned `sft_i2t.parquet` — about 2.9M mixed samples, covering the image captions inherited from Pretrain, reasoning-style Q&A on natural images, fine-grained Q&A on documents/charts, instructions synthesized by Gemini/Claude/GPT-4o, plus ~230K pure-text conversations (image column filled by an 8×8 black placeholder). Learning rate drops to ~5e-5, max sequence 768.

A common practice is to fully unfreeze the LLM during SFT, but this usually assumes a several-B-parameter base and a substantial amount of pure-text data mixed into SFT. MiniMind-V's language backbone is only 64M and ~92% of the current SFT data is image-related, so fully unfreezing the LLM would likely dilute its original general-language capability under the image-task gradients.

We therefore use `--freeze_llm 1`: **only the Projection and the first & last LLM layers are unfrozen, while the remaining N-2 layers keep their Pretrain weights**. The first layer is the first processing stage after visual tokens enter the LLM and thus bears the cross-modal fusion; the last layer shapes the format and style of the answer; the middle layers retain the knowledge from LLM Pretrain and are not overwritten by image-task gradients. The ~230K pure-text samples further act as a regularizer for general-language capability.

> Training Time and Loss Trend (for reference only)

On a single NVIDIA 3090, SFT takes ~2 hours per `epoch` in practice; dense and MoE finish in similar time (activated parameters are on the same order, with the gap mostly coming from the extra memory traffic of expert routing). Pretrain data volume is ~45% of SFT's, so one Pretrain epoch can be roughly scaled by that ratio. At a typical cloud price of ~1.5 RMB/hour for a 3090, a full SFT round costs about 3 RMB.

Pretrain [768+8] (dense & moe)  
![input](./images/pretrain_loss.jpg)

SFT [768+8] (dense & moe)  
![input](./images/sft_loss.jpg)

## Ⅲ Model Weights

| Format | ModelScope | HuggingFace |
|---|---|---|
| Native PyTorch (`*.pth`) | [minimind-3v-pytorch](https://www.modelscope.cn/models/gongjy/minimind-3v-pytorch) | [minimind-3v-pytorch](https://huggingface.co/jingyaogong/minimind-3v-pytorch) |
| Transformers | [minimind-v collection](https://modelscope.cn/collections/MiniMind-V-42b841dde22d41) | [minimind-v collection](https://huggingface.co/collections/jingyaogong/minimind-v-67000833fb60b3a2e1f3597d) |

> Note: The Transformers version is the `MiniMind-V` model after single-image SFT

# 📌 Test

### Effect Test

Prompt: `<image>\nPlease illustrate the image through your words.`

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
        <img src="./dataset/eval_images/image-01-golden-dog-balloons.jpg" alt="golden-dog">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>The image shows a golden-brown dog in a park setting with a clear blue sky above. The dog appears to be a golden-brown dog with a cheerful expression, which is characteristic of a Golden Bay breed. The dog's fur is a lighter color, contrasting with the golden balloon, which is likely a balloon with a bright yellow color. The dog's tail is bald, and its tail is darker, indicating it might be a wild, possibly a festive holiday. The dog's bright yellow balloon, along with its tail, stands out against the lighter background of the park. There are no other people or identifying marks in the immediate vicinity of the dog, emphasizing its celebration and a sense of joy.</td>
      <td>The image depicts a golden-brown dog with a golden-brown coat, likely a Border Collie, standing on a grassy field with a clear blue sky. The dog appears to be a German Shepherd, given its size and the positioning of its tail. Its ears are perked up, and it has a golden-brown coat with a lighter golden hue. The dog's eyes are looking forward, and it seems to be looking down at it. The sky is partly cloudy, suggesting a fair weather day. The dog is standing on a grassy field, with some trees and others in the distance, indicating a calm and natural setting. The overall mood of the image is one of tranquility and enjoyment.</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/image-02-rainbow-umbrella-street.jpg" alt="rainbow-umbrella">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>The image shows a rainy day with a rainbow-colored umbrella. The umbrella has a colorful mottled pattern of colorful, orange, yellow, red, green, purple, and blue. The umbrella is surrounded by a bright, irregular, bright daylight with irregular, warm lighting. The umbrella's bright colors range from pink, yellow, purple, green, blue, and red, creating a cool, wet atmosphere. The rainbow-colored umbrella is lit by the bright sky, indicating a sunny day. The umbrella is slightly warmer than the rainbow-colored umbrella, with a reflective surface and a blurred background that suggests a shaded area. The umbrella is in sharp focus, allowing the umbrella to be visible on the scene. The overall impression is one of a tropical or subtropical environment, with the rainbow-colored umbrella in a vibrant green color, creating a sense of depth and natural beauty.</td>
      <td>The image shows an outdoor scene with a rainy umbrella in the background, likely during a time of day when umbrellas are illuminated. The umbrellas are arranged in a turn, creating a dynamic and colorful umbrella. The umbrellas have a smooth, flat surface with some visible wrinkles and grouting, indicating a high level of detail. The umbrellas are capped with a slightly darker pink hue, providing a lighter gray color that contrasts with the umbrellas. They are positioned in the distance, adding to the umbrella's appearance. The umbrellas are the main subjects, and the umbrellas are positioned in the distance.</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/image-03-cherry-blossom-bike.jpg" alt="cherry-blossom-bike">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>The image depicts a vibrant pink bicycle, with its wicker basket attached to the bicycle. The bicycle's design is vibrant, with a sleek design that complements the vibrant colors of the pink and white flowers. The basket appears to be made of a material with a durable fabric, and it has a low-profile design, with a slight front wheel and a single rider, which are often used for pedestrian traffic. The bicycle's frame is not fully visible, but it has a black frame and a black handle. There are no visible people or distinctive features that suggest a specific location or event, such as a bike rack or a bike rack, which are not present in this image. The pink bicycle's body is partially visible and has a single seat, suggesting it is likely a bicycle. There are no discernible background elements that provide context or distraction from the image.</td>
      <td>The image depicts a bright pink bicycle, captured from a perspective view from an elevated perspective. The bicycle is an iconic symbol of the bike, with its rider sporting a touch of white, white, and a touch of purple. The bike is a traditional style with a simple bike and a basketball hoop. It's a sunny day, and the lighting suggests it might be daytime. The scene is set against a pastoral scene, with the pathways leading towards a pathway that casts a warm glow, indicating the serene and warm atmosphere of the scene. The colors in the image are vibrant, with the pink of the pathway standing out against the pale pink of the bicycle. There's a sense of tranquility and freedom of movement.</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/image-04-yellow-car.jpg" alt="yellow-car">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>The image shows a yellow sports car positioned on a road with a clear blue sky. The car is positioned on a road with a prominent yellow roof, which is characteristic of a yellow sports car. The car is positioned in the front of the car, facing the camera, with its front grille and lower body lines visible, which suggests a sporty aesthetic. The car is parked on a road that appears to be a high-performance sports car, and the background is a parking lot with a few small boats, which is a common feature in such vehicles. There are no visible windows or other objects in the immediate vicinity of the car.</td>
      <td>The image features a bright yellow sports car parked on a sidewalk. The car is a paved road with a road that appears to be a parking lot, and the sidewalk is bordered by a thin line of trees and a clear blue sky. There are no visible passengers, and the car is parked on a road with a few cars, indicating it is likely a busy day. The car has a sleek design with a high-performance lines and a large round headlamps.</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/image-05-superhero-rooftop.jpg" alt="superhero">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>The image shows a stylized silhouette of a character from a superhero film series. The character is wearing a long-sleeved blue suit with a blue collar and a red band on the left chest, paired with red trousers. The suit is equipped with a blue suit and a red band on top, and the suit is adorned with a red band, which is also in the midst of a superhero character. The suit is standing outdoors, and the character is on a city street, with a large, blue suit standing out. The sky is a gradient of warm colors, suggesting it is either dawn or dusk. The overall impression is one of a high-energy, urban nighttime scene, with a focus on the character and the character.</td>
      <td>The image depicts a spaceship scene at what appears to be a cityscape during what appears to be sunset. The scene is dominated by the sky, with the silhouettes of a figure standing on a raised platform, which is the central subject of the image. The figure is dressed in a red and blue suit with a white shirt, which suggests a style commonly associated with the "UNITED SUNDER" superhero franchise. The suit is primarily red with a red hue and a white shirt. There is a figure with a red belt on the right, holding a red object that could be a shield or a wrench. The person is standing on a street with a couple of buildings, indicating that the location might be a city with buildings that are part of the UNITED SUNDER. The sky is filled with a warm orange hue, suggesting a sunset. The overall atmosphere is one of dynamism and fantasy.</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/image-06-racecar-drift.jpg" alt="racecar">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>The image captures a dynamic scene at what appears to be a race. In the foreground, a red sports car is mounted on a racetrack, its front face is turned slightly to the left. The car has a prominent rear wing and aerodynamic elements, which suggests a high-speed speed and aerodynamic profile. The track is adorned with multiple windows, one of which is visible, allowing a view of the ground level. The background shows a crowd of spectators, some dressed in red, while others are standing, possibly in the air, in the distance, indicating that this is likely the race of the race. The lighting is warm, and the ground is dim, which gives the racing a warm ambiance. There are no people visible in the image, and the image is taken at a time when the race is being captured and the car is in the air. The lighting conditions suggest that it is either early morning or late afternoon, as the sun is low on the horizon, casting a warm glow and creating a dramatic effect on the track.</td>
      <td>The image captures a dynamic scene at what appears to be a high-speed race, likely during the golden hour, given the warm lighting and the soft shadows. The track is bathed in a soft glow, indicating the sun is high and possibly at the moment of the race. There are no visible smoke, orbits, or disturbances in the sky, suggesting the race is either during the golden hour or during the golden hour. The race is a sizable form of aerodynamic body, with the front wheel being more pronounced, indicating it may have been used for speed and possibly aerodynamics. The track is bordered by a blurred audience, which may suggest a focus on the race's grandeur or the driver's presence. The track itself has a glossy finish, with some areas showing darker lines and the surrounding structures, which could be part of a track or a similar event.</td>
    </tr>
  </tbody>
</table>

### Effect Summary:

Both models can identify the primary subject in most images (dog, umbrella, bicycle, sports car, superhero, racecar, etc.), but both exhibit repetitive phrasing and hallucinated details, placing overall performance at a stage of "understanding the gist but inaccurate on details".

Across these 6 samples, the MoE variant produces slightly richer scene descriptions and better color recognition (e.g. the cherry-blossom scene, rainbow umbrella, yellow sports car), while the Dense model tends to be more concise. Over a broader evaluation set, however, MoE occasionally suffers from more pronounced entity-level hallucinations or repetitive generation, showing higher variance than Dense — making Dense the safer default configuration.

Visual signals act as a special "foreign language" to the LLM, so the ceiling of "learning that language" is bounded by the LLM's own language ability. A stronger backbone extracts more value from the same image-text data; swapping MiniMind-V's backbone for a several-B-scale LLM yields clearly sharper details and more coherent reasoning.

#### Future Areas for Improvement:

```text
> Introduce dynamic resolution and Tile-based encoding (like LLaVA-NeXT) to break through the fixed resolution limit.
> Visual Encoder could be upgraded to stronger vision encoders for finer-grained image features.
> Extend multi-image understanding, video understanding, and Visual Grounding capabilities.
> ...
```

# 📌 Acknowledge

> [!TIP]
> If `MiniMind-V` is useful to you, a ⭐ on GitHub is welcome. <br/>
> Issues and PRs are the best place to share problems or improvements found while using the project.

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

