<div align="center">

![logo](./images/logo.png)

</div>


<div align="center">

![visitors](https://visitor-badge.laobi.icu/badge?page_id=jingyaogong/minimind-v)
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
  <h3>"大道至简"</h3>
</div>

<div align="center">

中文 | [English](./README_en.md)

</div>

* 此项目旨在从0开始，仅用3块钱成本 + 2小时！即可训练出65M参数的超小多模态视觉语言模型**MiniMind-V**。
* **MiniMind-V**最小版本体积仅为 GPT3 的约 $\frac{1}{2600}$，力求做到个人GPU也可快速推理甚至训练。
* **MiniMind-V**是[MiniMind](https://github.com/jingyaogong/minimind)纯语言模型的视觉能力额外拓展，同系列多模态Omni模型详见[MiniMind-O](https://github.com/jingyaogong/minimind-o)。
* 项目同时包含了VLM大模型的极简结构、数据集清洗、Pretrain、SFT等全过程代码。
* 这不仅是一个开源VLM模型的最小实现，也是入门视觉语言模型的简明教程。
* 希望此项目能为所有人提供一个抛砖引玉的示例，一起感受创造的乐趣！推动更广泛AI社区的进步！

> 注：本项目基于 Apache 2.0 协议开源，完全免费。“2 小时” 指 SFT 阶段在单张 NVIDIA 3090 上跑完 `1 epoch` 的实测耗时，“3 块钱” 指对应时段的 GPU 租用成本。



<div align="center">

![minimind-3v](./images/minimind-3v.gif)

[🔗🤖在线体验](https://www.modelscope.cn/studios/gongjy/MiniMind-V) | [🔗🎞️视频介绍](https://www.bilibili.com/video/BV1Sh1vYBEzY)

</div>

# 📌 项目介绍

“用乐高拼出一架飞机，远比坐在头等舱里飞行更让人兴奋！”
构建VLM范式的多模态大模型是否真的如想象中那样复杂？它的代码实现到底如何？
训练过程究竟难不难？那么现在，探索它们的答案，一起感受创造的乐趣吧！

> [!TIP]
> （截至2026-04-20）MiniMind-V 系列已完成了以下型号模型训练，最小仅需65M (0.065B)，即可具备识图和对话的能力！

| 模型 | 参数量 | Release |
|---|---|---|
| minimind-3v-moe | 200M-A65M | 2026.04.20 |
| minimind-3v | 65M | 2026.04.20 |
| MiniMind2-V | 104M | 2025.02.20 |
| MiniMind2-Small-V | 26M | 2025.02.20 |
| minimind-v-v1-small | 27M | 2024.10.04 |
| minimind-v-v1 | 109M | 2024.10.04 |

#### 👉 更新日志

<details>
<summary> <b>2026-04-20</b> </summary>

- 更新模型检查点：minimind-3v (65M) / minimind-3v-moe (200M-A65M)
- Projector 更新：添加 `LayerNorm`，去掉 reshape token 合并（P32 原生 64 token，无需下采样）
- Vision Encoder 换为 `SiglipVisionModel`（P32，固定 256×256）
- 训练数据切到 ALLaVA-4V（Pretrain 127 万 / SFT 290 万，已合并为单阶段 SFT）
- Freeze 策略更新：`freeze_llm=1` 解冻首末两层，Pretrain/SFT 默认改为 `2`/`1`；`max_seq_len` 360 → 450
- 其他 bugfix 与小调整

</details>

<details> 
<summary> <b>2026-04-01</b> </summary>

- 新增 minimind-3v (67M) 和 minimind-3v-moe (201M-A67M) 模型
- 统一使用768+8架构，支持dense和moe两种模式
- 视觉编码器从CLIP切换为SigLIP2（siglip2-base-p16-256-ve）
- 投影模块从QFormer改为MLP Projection + reshape压缩
- 数据集格式更新为parquet，混合数据源、更新tokenizer，图像占位符改为`<|image_pad|>`、新增WebUI：支持动态扫描模型目录、下拉菜单切换模型
- 模型代码重构，LLM/VLM统一适配Transformers格式
- 训练脚本支持DDP多卡、bfloat16混合精度、torch.compile加速

</details>

<details> 
<summary> <b>2025-10-24</b> </summary>

- bug修复：模型权重不对应
- 适配[「minimind-1024更新」](https://github.com/jingyaogong/minimind)
- 代码重构：训练和评估脚本规范化
- 新增完整的断点续训支持

</details>

<details>

<summary> <b>More...</b> </summary>

**2025-04-27**

- 兼容性更新
- 适配[「minimind仓库新特性」](https://github.com/jingyaogong/minimind/issues/370)
- 规范化部分代码

**2025-02-20**

- MiniMind2-V伴随MiniMind2同步更新
- 大幅减少所有冗余代码，规范代码格式
- 大幅精简模型冗余结构
- 更新数据集格式，拓展新的SFT数据集
- 比前代VLM更优秀的效果！

**2024-10-05**

- MiniMind-V如期而至，首次开源

</details>

# 📌 快速开始

<details>
<summary>分享本人的软硬件配置（仅供参考）</summary>

* CPU: Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz
* RAM: 128 GB
* GPU: NVIDIA GeForce RTX 3090(24GB) * 8
* Ubuntu==20.04
* CUDA==12.2
* Python==3.10
* [requirements.txt](./requirements.txt)

</details>

## 第0步（必须）

### 1' 环境准备

```bash
# 克隆仓库代码
git clone --depth 1 https://github.com/jingyaogong/minimind-v
# 安装必要依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2' 下载资源

```bash
# 下载 SigLIP2 视觉编码器到 ./model/siglip2-base-p32-256-ve
modelscope download --model gongjy/siglip2-base-p32-256-ve --local_dir ./model/siglip2-base-p32-256-ve
# 下载 MiniMind 语言模型权重到 ./out 目录下（作为训练 VLM 的基座语言模型）
modelscope download --model gongjy/minimind-3v-pytorch llm_768.pth --local_dir ./out
```

注：也可从 [ModelScope Collection](https://modelscope.cn/collections/gongjy/MiniMind-V) 或 [HuggingFace Collection](https://huggingface.co/collections/jingyaogong/minimind-v-67000833fb60b3a2e1f3597d) 选择对应内容 `git clone`（需 LFS）下载，此处不再赘述。

完成后，结构应如下：

```text
minimind-v/
├── model/
│   ├── siglip2-base-p32-256-ve/
│   └── ...
├── out/
│   └── llm_768.pth
└── ...
```

## Ⅰ 🚀 模型推理

### 1' 下载发布权重

```bash
# 下载发布权重到 ./out 目录下
modelscope download --model gongjy/minimind-3v-pytorch --local_dir ./out
```

### 2' 命令行问答

```bash
# load_from='model': 加载原生PyTorch权重, load_from='其他路径': 加载transformers格式
python eval_vlm.py --load_from model --weight sft_vlm
```

如果使用 transformers 格式模型，可先下载模型目录：

```bash
git clone https://huggingface.co/jingyaogong/minimind-3v
python eval_vlm.py --load_from minimind-3v
```

### 3' 启动 WebUI（可选）

```bash
# ⚠️ 须先将 transformers 格式模型文件夹复制到 ./scripts/ 目录下，web_demo_vlm 脚本会自动扫描该目录下包含权重文件的子文件夹，如不存在则报错
cp -r minimind-3v ./scripts/minimind-3v
cd scripts && python web_demo_vlm.py
```

## Ⅱ 🛠️ 模型训练

<details style="color:rgb(128,128,128)">
<summary>注：提前测试Torch是否可用cuda</summary>

```python
import torch
print(torch.cuda.is_available())
```

如果不可用，请自行去 [torch_stable](https://download.pytorch.org/whl/torch_stable.html) 下载 whl 文件安装。

</details>

### 1' 下载数据

快速开始时，直接从[数据集链接](https://huggingface.co/datasets/jingyaogong/minimind-v_dataset)下载 `sft_i2t.parquet`，并放到 `./dataset` 下即可。

<details style="color:rgb(128,128,128)">
<summary>注：数据集须知</summary>

【注1】之前需解压50万零碎的图像文件可能非常慢。2025-12-27起，数据集格式统一为 Parquet，图文一体化存储，体积更小，无需解压，加载更快。

【注2】Parquet 是列式存储格式，支持高效压缩和快速读取。如果你对它感到陌生，可以预览数据内容，在 `dataset/` 目录下执行 `python lm_dataset.py` 可视化前5条图文对。

Pretrain 数据（可选；仅包含 caption 子集）：
```bash
wget https://hf-mirror.com/datasets/jingyaogong/minimind-v_dataset/resolve/main/pretrain_i2t.parquet -P ./dataset
```

SFT 单文件 290 万条已把 Pretrain 作为子集合并，经全局 dictionary encoding 去重后体积只比 SFT 原版多 ~10%，可覆盖所有训练阶段。因此快速复现时可以跳过 Pretrain，直接进入 SFT。

</details>

### 2' 开始训练

推荐直接执行 SFT。默认 `--freeze_llm 1`，即训练 `vision_proj` 和 LLM 首尾层，保留中间层原有语言能力：

```bash
python train_sft_vlm.py --epochs 2 --from_weight llm
```

如果希望让 Projector 先完成一轮图文对齐，再进入 SFT，可额外执行 Pretrain：

```bash
python train_pretrain_vlm.py --epochs 2 --from_weight llm
python train_sft_vlm.py --epochs 2 --from_weight pretrain_vlm
```

执行完成后，`out/` 下会生成 `sft_vlm_*.pth` 作为 SFT 权重。

<details style="color:rgb(128,128,128)">
<summary>注：训练须知</summary>

- 支持断点续训：添加`--from_resume 1`参数可从上次中断处继续训练
- 支持GPU数量变化：续训时GPU数量改变会自动转换step
- 原子性保存：使用临时文件+替换机制，防止保存过程中断导致权重损坏
- 每次保存同时生成`out/**.pth`（模型权重）和`checkpoints/**_resume.pth`（训练状态）文件

```bash
# 训练中断后，使用相同命令并添加 --from_resume 1
python train_sft_vlm.py --epochs 4 --from_resume 1
```

**参数说明：**
- `--from_weight`: 基础权重名称（llm, pretrain_vlm, none等）
- `--save_weight`: 保存权重的前缀名
- `--from_resume`: 是否续训（0=从头开始，1=从检查点继续）
- `--freeze_llm`: 冻结策略（0=全参可训，1=proj + LLM 首尾层，2=仅训 proj）。Pretrain 默认 2，SFT 默认 1
- 更多可直接参考代码

</details>

### 3' 测试已训练模型（可选）

确保需要测试的模型 `*.pth` 文件位于 `./out/` 目录下。
也可以直接去[此处](https://huggingface.co/jingyaogong/minimind-3v-pytorch)下载使用我训练的`*.pth`文件。

```bash
# 测试SFT模型（默认）
python eval_vlm.py --weight sft_vlm

# 测试Pretrain模型
python eval_vlm.py --weight pretrain_vlm
```

---

> [!TIP]
> 训练脚本均为 PyTorch 原生框架，均支持多卡加速，假设你的设备有 N (N＞1) 张显卡：

单机N卡启动训练方式 (DDP, 支持多机多卡集群)

```bash
torchrun --nproc_per_node N train_xxx.py
```

<details>
<summary>注：其它须知</summary>

<del>
单机N卡启动训练 (DeepSpeed)

```bash
deepspeed --master_port 29500 --num_gpus=N train_xxx.py
```
</del>

可根据需要开启wandb记录训练过程

```bash
# 需要登录: wandb login
torchrun --nproc_per_node N train_xxx.py --use_wandb
# and
python train_xxx.py --use_wandb
```

通过添加`--use_wandb`参数，可以记录训练过程，训练完成后，可以在wandb网站上查看训练过程。通过修改`wandb_project`
和`wandb_run_name`参数，可以指定项目名称和运行名称。

【注】：25年6月后，国内网络环境无法直连WandB，MiniMind项目默认转为使用[SwanLab](https://swanlab.cn/)作为训练可视化工具（完全兼容WandB API），即`import wandb`改为`import swanlab as wandb`即可，其他均无需改动。

</details>

# 📌 模型细节

MiniMind-V 的语言主干即孪生项目 [minimind](https://github.com/jingyaogong/minimind) 训练得到的 `llm_768.pth`，LLM 本身的结构、训练细节与实验分析不在本仓库重复，默认读者对 MiniMind LLM 已有基本了解。未接触过也不影响照着“快速开始”跑通 MiniMind-V，流程自洽。

顶部 “从 0 训练” 和 “65M” 两个简化口号在这里也需要说明。“从 0 训练” 指 VLM 本身从零构建（Projection 随机初始化、LLM 首末层微调对齐），但 LLM 主干并非从零 pretrain，而是基于 MiniMind 的语言模型权重续训而来；若要严格意义上的 “完全从零 pretrain”，请先在 MiniMind 训一版 LLM 再迁回本项目。“65M” 指可训练部分的主干规模（LLM ~64M + Projection ~1M）；视觉编码器 SigLIP2 另有 ~95M 参数全程冻结、仅作图像特征提取，因此推理时整机参数量约 160M（dense）/ 294M（MoE）。

VLM 在 LLM 基础上增加 Visual Encoder 和特征投影两个子模块，引入模态混合分支以支持多模态输入：
![LLM-structure](./images/VLM-structure.jpg)
![LLM-structure](./images/VLM-structure-moe.jpg)


<details>
<summary> 【重要】一些有趣的思考 </summary>

此处不妨展开想一想两个问题：

* 什么叫做**L**arge **L**anguage **M**odel (LLM)？
* 什么叫做多模态模型？

[这篇文章](https://www.jiqizhixin.com/articles/2024-09-15-3)完美吻合本人的想法：
大语言模型（LLM）名字虽然带有语言二字，但它们其实与语言关系不大，这只是历史问题，更确切的名字应该是自回归 Transformer
或者其他。LLM 更多是一种统计建模的通用技术，它们主要通过自回归 Transformer 来模拟 token 流，而这些 token
可以代表文本、图片、音频、动作选择、甚至是分子等任何东西。
因此，只要能将问题转化为模拟一系列离散 token 的流程，理论上都可以应用 LLM 来解决。
实际上，随着大型语言模型技术栈的日益成熟，我们可能会看到越来越多的问题被纳入这种建模范式。也就是说，问题固定在使用 LLM
进行『下一个 token 的预测』，只是每个领域中 token 的用途和含义有所不同。

[ZJU-LiXi老师](https://person.zju.edu.cn/xilics#694283)同样谈及过类似观点（原话大意如下）：
文本、视频、语音、动作等在人类看来属于「多模态」信号，但所谓的「模态」其实只是人类在信息存储方式上的一种分类概念。
就像`.txt`和`.png`文件，虽然在视觉呈现和高级表现形式上有所不同，但它们本质上并没有根本区别。
之所以出现「多模态」这个概念，仅仅是因为人类在不同的感知层面上对这些信号的分类需求。
然而，对于机器来说，无论信号来自何种「模态」，最终它们都只是以一串二进制的「单模态」数字序列来呈现。
机器并不会区分这些信号的模态来源，而只是处理和分析这些序列背后所承载的信息内容。

个人认为**G**enerative **P**retrained **T**ransformer (GPT) 比 **L**arge **L**anguage **M**odel (LLM)更为贴切，
因此本人表达上更习惯用"GPT"去代表LLM/VLM/类GPT架构的系列模型，而非为了蹭OpenAI的热度。

至此，我们可以用一句话总结GPT的所作所为：

GPT模型根据现有token预测输出下一个下下一个下下下一个token ...，直到模型输出结束符；此处的"token"其实并不需要一定是文本！

```text
> 对于LLM模型，如果需要理解"图片"，我们只要把"图片"作为对一种特殊的从来没见过的"外国语言"，通过"外语词典"翻译后即可作为特殊的语言输入LLM
> 对于LLM模型，如果需要理解"音频"，我们只要把"音频"作为对一种特殊的从来没见过的"外国语言"，通过"外语词典"翻译后即可作为特殊的语言输入LLM
> ...
```

<u>**为了得到MiniMind-V，我们只需要完成这2件事即可：**</u>

1. 借助擅长翻译图片的 **"外语词典"** ，把图片从 **"外国语言"** 翻译为模型便于理解的 **"LLM语言"**
2. 训练微调LLM，使其和 **"外语词典"** 度过磨合期，从而更好的理解图片

"外语词典" 称之为Visual Encoder模型。
和LlaVA、Qwen-VL等视觉语言模型类似，MiniMind-V当前选用开源SigLIP2系列模型作为Visual Encoder。
具体使用[siglip2-base-p32-256-ve](https://huggingface.co/jingyaogong/siglip2-base-p32-256-ve)，
一种基于 ViT-B/32 架构的Visual Encoder用于描述图像文本信息。
当前使用的 SigLIP2 NaFlex 视觉编码器会根据预处理结果生成 64 个 patch token（256×256 图像 / patch_size 32 = 8×8 = 64）作为 encoder 编码层的输入，
最终产生1×768维的嵌入向量用于和文本对计算误差。
我们并不需要最终嵌入表示，因此只取encoder层的输出，也就是VIT核心主干的输出特征即可。
它拿到前一层 64×768 大小的特征，经过 LayerNorm + 2 层 MLP（Linear→GELU→Linear）投影到 LLM 的隐藏维度，最终作为 64 个 visual token 输入 MiniMind-V——这一步完成的就是跨模态特征对齐：让原生视觉特征进入文本 token 所在的语义空间，使两者可以在同一空间里交互。

[LlaVA-1](https://arxiv.org/pdf/2304.08485)使用简单的线性变换完成对齐，[LlaVA-1.5](https://arxiv.org/pdf/2310.03744)升级为2层MLP，MiniMind-V采用与LlaVA-1.5相同的MLP Projection方案（P32 原生输出 64 token，无需额外 reshape 压缩）。

![llava-structure](./images/llava-structure.png)

MiniMind-V的主要结构已介绍完毕。

</details>


---

下面，我们简单讨论MiniMind-V的外部输入输出的变化。

VLM的输入依然是一段文本，其中包含特殊的`<image>`占位符。
在计算文本嵌入后，可以将图像编码器生成的向量投影到该占位符对应的嵌入部分，替换掉原先的占位符embedding。
例如：

```text
<image>\n这个图像中有什么内容？
```

在`minimind-v`中，使用64个`<|image_pad|>`组成的占位符代替图像（SigLIP2 P32 直接输出 64 个 patch token，经 MLP 投影为 64 个 visual token），因此`minimind-v`的prompt为：

```text
<|image_pad|><|image_pad|>...<|image_pad|>(×64)\n这个图片描述的是什么内容？
```

计算完embedding和projection，用视觉特征替换掉对应占位符的embedding后，整个计算过程到输出则和LLM部分没有差异。

![input](./images/minimind-v-input.jpg)

至此，`MiniMind-V`的所有细节呈现完毕，VLM模型子类继承自`MiniMind`，仅做**最小**变更而产生，核心算法改动`< 50行`，迁移难度极低，和`LlaVA`等模型具体实现存在区别，但思路一致。

# 📌 实验

## Ⅰ 数据集

本轮训练用到的图文数据全部来自 [ALLaVA-4V](https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V) 系列。
相比以往从几份 LLaVA 衍生集拼接得到的数据，ALLaVA-4V 的质量更整齐、中英双语原生对照，细粒度描述也更充分。
它由两个子源构成：一份是 LAION 里挑出来的高质量图片（自然图像为主），一份是 VFLAN 指令流里挑出来的图片（文档、图表、合成场景居多）。

- **Pretrain（`pretrain_i2t.parquet`，约 127 万条 / ~64 万张唯一图像）**
  - `ALLaVA-Caption-LAION-4V` 英/中：~47万 + ~44万
  - `ALLaVA-Caption-VFLAN-4V` 英/中：~19万 + ~17万
  - 任务形式为"请描述这张图片"类的单轮长描述，用于让模型建立视觉 token 到语言 token 的基础对齐。

- **SFT（`sft_i2t.parquet`，约 290 万条 / ~65 万张唯一图像）**
  - `ALLaVA-Instruct-LAION-4V` 英/中：~47万 + ~47万
  - `ALLaVA-Instruct-VFLAN-4V` 英/中：~19万 + ~17万
  - `Instruct-LAION-4v-gemini-claude-ensembled`（Gemini/Claude 合成的增广指令）：~5万
  - `Instruct-LAION-4oiterative`（基于 GPT-4o 迭代润色的指令）：~5万
  - 纯文本对话（保留基础语言能力，图像列填 8×8 黑图占位）：~23万
  - **Pretrain caption 数据全量合并**（与 pretrain 同源、~99% 图像重叠）：~127万
  - 任务形式混合了"围绕图片的推理式问答"、"caption 长描述"和"纯文本对话"，既考验细节追问/长链条推理，也兼顾图像描述与通用语言能力。

合计约 290 万条样本，Pretrain 阶段可直接跳过（SFT 已把 Pretrain 作为子集吸收）。中英比例大致均衡。
考虑到 MiniMind-V 的语言主干仅 65M，把英文和中文一起喂进去是比较稳妥的做法——中文语料对母语输出更友好，而英文原生描述通常更精细，两者互为补充。

图像统一 `resize` 到 **256×256**（与 SigLIP2 NaFlex 编码器的输入规格相对应，P32 产生 64 个 patch token），重新编码为 JPEG 打包进 parquet。

(`pretrain_i2t.parquet`) Pretrain 数据集格式：

```text
列名: conversations (json string), image_bytes (binary)

conversations 示例:
[
  {"role": "user", "content": "<image>\n请提供对图片的详细文字描述。"},
  {"role": "assistant", "content": "这张图片展示的是一个..."}
]
image_bytes: <图像二进制数据>
```

(`sft_i2t.parquet`) 单图 SFT 数据集格式：

```text
列名: conversations (json string), image_bytes (binary)

conversations 示例:
[
  {"role": "user", "content": "根据图片推断这个场景的大致时间？<image>"},
  {"role": "assistant", "content": "从光线和阴影来看..."}
]
image_bytes: <图像二进制数据>
```

> 注：sft_i2t.parquet 共约 290 万条数据，其中约 140 万条为图像 instruct 对话、约 127 万条为图像 caption 描述（Pretrain 数据合并而来）、约 23 万条为纯文本对话（t2t，图像列填 8×8 黑图占位，用于保持模型的基础语言能力）。由于 Pretrain 已作为子集并入，可选择跳过 Pretrain 阶段直接进行 SFT。

数据集下载地址：([ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind-v_dataset) | [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind-v_dataset))

## Ⅱ 训练

训练分为两个阶段（Pretrain 可选；SFT 必需），均冻结 Visual Encoder 梯度，仅训练 Projection 和部分 LLM 层。
训练基于 LLM Pretrain 权重初始化，支持 DDP 多卡训练、混合精度（bfloat16）、torch.compile 加速和 swanlab 日志记录。

> train_pretrain_vlm（可选）

Pretrain 阶段从约 127 万条图文描述数据中学习图片的通用知识（如鹿是鹿，狗是狗）。
此阶段采用较高学习率（~4e-4），最大序列长度 450，**完全冻结 LLM 和 Visual Encoder，仅训练 Projection**（`--freeze_llm 2`）。
目的是让 Projector 干净地完成视觉 token 到语言 token 的基础对齐，不扰动 LLM 原有权重。
由于 SFT 阶段数据已把 Pretrain 全部样本包含为子集，此步骤可选，跳过可节省时间，但**先跑一轮 Pretrain 让 Projector 先行对齐，SFT 收敛更稳**。

> train_sft_vlm

SFT 阶段的数据即前述 `sft_i2t.parquet`，约 290 万条混合样本，涵盖 Pretrain 继承而来的图文 caption、自然图像上的推理问答、文档/图表的细节问答、Gemini/Claude/GPT-4o 合成的指令，以及约 23 万条纯文本对话（图像列填 8×8 黑图占位）。学习率降到 ~5e-6，最大序列 768。

常见做法是在 SFT 阶段把 LLM 全参解冻，但这通常建立在底座有几 B 参数、且 SFT 数据中混入大量纯文本这两个前提之上。MiniMind-V 的语言主干仅 64M，而当前 SFT 数据中 ~92% 与图像有关，若全参解冻，LLM 原有的通用语言能力极易被图文任务的梯度稀释。

这里采用 `--freeze_llm 1`：**只解冻 Projection 与 LLM 的首、末层，其余 N-2 层保持 Pretrain 时的权重**。首层是视觉 token 进入 LLM 后的第一道处理，直接承担跨模态融合；末层影响回答的格式与风格；中间层保留 LLM Pretrain 的知识，不被图文任务的梯度改写。那 23 万条纯文本样本对通用语言能力也起到类似正则的作用。

> 训练时间和 Loss 走势（仅供参考）

单张 NVIDIA 3090 上，SFT 跑完 `1 epoch` 实测约 2 小时，dense 与 MoE 用时接近（激活参数规模一致，差异主要来自 expert routing 的额外访存）；Pretrain 数据量约为 SFT 的 45%，单 epoch 耗时按比例粗估即可。按云上 3090 约 1.5 元/小时的行情，SFT 单轮成本落在 3 元上下。

Pretrain [768+8] (dense & moe)
![input](./images/pretrain_loss.jpg)

SFT [768+8] (dense & moe)
![input](./images/sft_loss.jpg)

## Ⅲ 模型权重

| 模型格式 | ModelScope | HuggingFace |
|---|---|---|
| 原生PyTorch (`*.pth`) | [minimind-3v-pytorch](https://www.modelscope.cn/models/gongjy/minimind-3v-pytorch) | [minimind-3v-pytorch](https://huggingface.co/jingyaogong/minimind-3v-pytorch) |
| Transformers 格式 | [minimind-v collection](https://modelscope.cn/collections/MiniMind-V-42b841dde22d41) | [minimind-v collection](https://huggingface.co/collections/jingyaogong/minimind-v-67000833fb60b3a2e1f3597d) |

> 注：Transformers版本均为单图 SFT 后的`MiniMind-V`模型

# 📌 评估

### 效果测试

Prompt: `<image>\n请描述这张图中的主要物体和场景。`

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
      <td>这张图中的主要物体是一只金毛寻回犬，以其金色的外套和友好的表情而闻名。金毛寻回犬以活泼的表情和友好而闻名，它们正在进行充满活力和活跃的互动。它的姿势和表情传达了一种快乐和积极的情绪。这只金毛寻回犬被描述为一个活泼的、充满活力的、充满活力的狗，在阳光明媚的户外环境中展现出来。它的眼睛是友好的，嘴巴是闭着的，这表明它在享受阳光明媚的日子里感到快乐和满足。在背景中，可以看到一片草地和晴朗的天空，暗示着这是一个宁静的户外环境。背景中的天空是淡蓝色的，有几朵零星的云，暗示着一个晴朗的日子。总体而言，这只金毛寻回犬的外表和活动充满了快乐和积极的情绪，使其成为一个充满活力和快乐的快乐时刻。</td>
      <td>这张图片展示了一只金毛寻回犬在草地上享受快乐。狗狗的毛色是丰富的棕色，带有白色和黄色的斑纹，它的耳朵竖立，嘴巴张开，看起来很快乐。它戴着一条项圈，表明它是一只宠物。背景是晴朗的蓝天，零星散落的云彩，表明照片是在一个阳光明媚的日子拍摄的。在狗狗的背后，有一片草地和几朵零星的云朵，表明这是一个宁静的自然环境。</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/image-02-rainbow-umbrella-street.jpg" alt="rainbow-umbrella">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>这张图中的主要物体是一把雨伞，根据其颜色和光泽，它是雨伞的特征之一。雨伞设计用于防止雨水进入，同时保持其透明度和可见性。它主要是绿色、红色、黄色和蓝色，还有一些白色的装饰，这在雨伞的背景下很常见。雨伞的反射和光线暗示着它可能是在一个有阳光的户外环境中拍摄的。雨伞是雨水的典型特征，表明它可能是雨天或湿润的天气条件。</td>
      <td>这张图片展示了一把雨伞，放在湿润的表面上，可能是一条街道上。这把雨伞有着醒目的彩虹色图案，颜色有各种蓝色、红色、黄色和紫色，形成了一种彩虹般的效果。雨伞的边缘略微弯曲，表明它要么是从雨滴中滴落，要么是从一个有反射表面的表面上滴落。雨伞本身是黑色的，有光泽，表明它已经湿润了。背景中可以看到一个城市环境，有建筑物和街道，天空多云，暗示着阴天或阴凉的一天。场景中看不到人，暗示着静止或有静止的迹象。</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/image-03-cherry-blossom-bike.jpg" alt="cherry-blossom-bike">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>这张图中的主要物体是一辆自行车，具体来说是一辆青绿色的自行车。这辆自行车是图片的中心焦点，展示了它的设计和特点。自行车的座椅是鲜艳的蓝色，带有白色座椅，座椅和座椅上点缀着粉色花朵，与青绿色的背景形成鲜明对比。自行车的前部配有黑色轮胎，胎部和车轮上还有小的装饰，可能是为了增加了视觉吸引力。背景中可见一片水域，可能是湖泊或河流，这表明照片是在一个宁静的自然环境中拍摄的，很可能是在一个水域旁边，因为远处可以看到一条蜿蜒的河流。天空阴沉，光线柔和，没有明显的阴影，这表明照片是在太阳处于较低角度的时候拍摄的。</td>
      <td>这张图片展示了一辆停在路边的古董自行车。自行车是鲜艳的蓝色，配有白色的车架和黑色的链条。它配有黑色的车把，表明它是一辆自行车。自行车上有两个踏板，一个位于左侧，一个位于右侧，都被遮挡着。背景是一片郁郁葱葱的绿色植被，暗示着这个地点可能靠近大自然或公园。天空晴朗，暗示着良好的天气条件。自行车上有黑色的座椅，与整体的蓝色和白色调相得益彰。自行车后面是郁郁葱葱的绿树，树叶浓密，与背景的宁静和郁郁葱葱形成鲜明对比。</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/image-04-yellow-car.jpg" alt="yellow-car">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>这张图中的主要物体是一辆现代高性能跑车。它以醒目的黄色为主色调，配有黑色轮毂和银色合金轮毂。车身设计时尚，前保险杠设计时尚，配有流线型的后扰流板，并配备了高性能轮胎。车轮是黑色的，带有明显的辐条，与黄色车身形成鲜明对比。背景中是晴朗的蓝天，偶尔有几朵零星的云，暗示着一个宁静的时刻。天空大部分晴朗，偶尔有几朵云，表明天气条件良好。没有可见的品牌标志或标识，使图片显得独特而不受欢迎。</td>
      <td>这张图片展示了一辆黄色的豪华跑车，停在一条道路上。这辆车是黄色的，具有独特的设计，前大灯设计，中央的LED大灯和尾灯，呈现出光滑的黄色。这辆车停在一个平坦的道路上，远处可见大海，天空是湛蓝色，零星云朵。背景中有大海和海岸线，天空晴朗，零星云朵，表明这是一个阳光明媚的日子。没有明显的特征表明这辆车是新的或正在进行的，而非最近的。</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/image-05-superhero-rooftop.jpg" alt="superhero">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>这张图中的主要物体是一位超级英雄的个人，被描绘在黄金时刻，这可以从他的姿势和服装上看出。超级英雄以超人般的姿势站立，一条腿抬起，脚踩在空中，另一条腿弯曲，仿佛在进行着一场致命的战斗。他的服装是蓝色的，带有红色的装饰，暗示着一种奔跑的姿态。他的头发是黑色的，右手拿着一个类似超级英雄的标志，左手拿着一个红色的锁。背景是一片模糊的天空，有云彩和远处的城市景观，暗示着一个城市环境。天空被从顶部的深蓝色渐变到地平线附近的暖色调，增强了日落的色彩。</td>
      <td>这张图片展示了一个穿着超级英雄服装的男性角色。他被抬起在一个巨大的、圆形的金属结构上，类似于一个巨大的、流线型的舞台，与他的服装形成鲜明对比。他的面部表情严肃而紧张，他直视着摄像机，给人一种紧张和戏剧性的印象。他的服装包括一件蓝色和红色的卫衣，与他穿的服装形成鲜明对比。他的服装配有金色的装饰，包括一件红色的衣领和一条腰带，与他的服装相得益彰。背景显示了一个城市天际线，有各种高楼大厦，暗示着一个城市环境。天空从地平线处温暖的金黄色过渡到地平线附近温暖的黄色，表明这是日落或日出。</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/image-06-racecar-drift.jpg" alt="racecar">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>这张图中的主要物体是一辆现代的赛车，具体来说是赛车。赛车是赛车运动的典型特征，具有高速、长长的后扰流板、高翼配置和独特的尾灯设计。赛车设计用于高速操控和高速操控，在赛道上形成一道弧线，增加了赛车的动感。赛车配备了空气动力学和空气动力学，这些特点在赛道上可见，表明这辆车可能是一辆高性能赛车。背景中可以看到赛车和观众，表明这辆赛车是高性能赛车的典型，而赛车的细节和风格表明这是一辆高性能赛车。赛车和车手身上没有可见的个人物品，使其成为图片的焦点。</td>
      <td>这张图片展示了一辆赛车在一条弯曲的赛道上。这辆车是一辆红色的赛车，具有流线型的车身，侧面醒目地显示着"NEWS"字样，暗示着这是一辆专业赛车。这辆车配备了赛车头盔，表明它是为高速赛车而设计的，因为能够提供保护。车轮和车身装饰有明显的红色装饰，包括一个大大的圆形标志，可能代表车队或品牌。背景中可以看到观众，表明这是一场赛车活动。天空晴朗，有几朵云，表明天气条件良好。赛道上没有明显的交通规则或安全措施，这表明这辆车是为赛车活动设计的。</td>
    </tr>
  </tbody>
</table>

### 效果小结：

两个模型在 6 张样本上均能正确识别主体对象（狗、雨伞、自行车、跑车、超级英雄、赛车），主体识别率均为 6/6，但仍伴随部分重复表述与细节幻觉，整体处于"能看懂大意、细节不够精准"的阶段。

MoE 版本的场景描述更丰富，对背景环境（城市街道、城市天际线、日落渐变色）和物体细节（彩虹色图案、蓝红服装配色、赛车车身标识）的捕捉优于 Dense。Dense 版本在表达简洁性上占优，重复表述更少。两者的幻觉程度接近，均会在局部细节上出现不准确的描述。

视觉信号对 LLM 而言相当于一门特殊的"外语"，"学外语"的上限因此受制于 LLM 自身的语言能力。底座越强，同样的图文数据喂进去收益越大；把 MiniMind-V 的主干换成几 B 量级的 LLM，细节准确度与推理连贯性的提升会相当明显。

#### 未来值得改进的方面：

```text
> 可引入动态分辨率和Tile-based编码（如LLaVA-NeXT），突破固定分辨率限制。
> Visual Encoder可升级为更强的视觉编码器，获取更细粒度的图像特征。
> 拓展多图理解、视频理解和视觉定位（Visual Grounding）能力。
> ...
```

# 📌 致谢

> [!TIP]
> 如果您觉得 `MiniMind-V`对您有所帮助，可以在 GitHub 上加一个⭐<br/>
> 水平有限难免存在未知的纰漏，欢迎所有人在Issues交流指正或提交PR改进项目<br/>
> 您的支持就是持续改进项目的动力，谢谢！

## 🤝[贡献者](https://github.com/jingyaogong/minimind-v/graphs/contributors)

<a href="https://github.com/jingyaogong/minimind-v/graphs/contributors">
  <img width="200" src="https://contrib.rocks/image?repo=jingyaogong/minimind-v" />
</a>

## 😊鸣谢

<a href="https://github.com/xinyanghuang7"><b>@xinyanghuang7</b></a>: <a href="https://github.com/xinyanghuang7/minimind-v/tree/hxy">多图vlm分支</a> | <a href="https://github.com/jingyaogong/minimind-v/tree/32cf4c5c01337231fd907b92d513de8945594263">仓库截至此版本提供</a> 

<details> 
<summary> <b>参考链接 & 感谢以下优秀的论文或项目</b> </summary>

- 排名不分任何先后顺序
- [LlaVA](https://arxiv.org/pdf/2304.08485)
- [LlaVA-VL](https://arxiv.org/pdf/2310.03744)
- [Chinese-LLaVA-Vision-Instructions](https://huggingface.co/datasets/LinkSoul/Chinese-LLaVA-Vision-Instructions)

</details>

## 🫶支持者

<a href="https://github.com/jingyaogong/minimind-v/stargazers">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://bytecrank.com/nastyox/reporoster/php/stargazersSVG.php?user=jingyaogong&repo=minimind-v&theme=dark"/>
      <source media="(prefers-color-scheme: light)" srcset="https://bytecrank.com/nastyox/reporoster/php/stargazersSVG.php?user=jingyaogong&repo=minimind-v"/>
      <img alt="github contribution grid snake animation" src="https://bytecrank.com/nastyox/reporoster/php/stargazersSVG.php?user=jingyaogong&repo=minimind-v&theme=dark"/>
    </picture>
</a>

<a href="https://github.com/jingyaogong/minimind-v/network/members">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://bytecrank.com/nastyox/reporoster/php/forkersSVG.php?user=jingyaogong&repo=minimind-v&theme=dark"/>
      <source media="(prefers-color-scheme: light)" srcset="https://bytecrank.com/nastyox/reporoster/php/forkersSVG.php?user=jingyaogong&repo=minimind-v"/>
      <img alt="github contribution grid snake animation" src="https://bytecrank.com/nastyox/reporoster/php/forkersSVG.php?user=jingyaogong&repo=minimind-v&theme=dark"/>
    </picture>
</a>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=jingyaogong/minimind-v&type=Date&theme=dark"/>
  <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=jingyaogong/minimind-v&type=Date"/>
  <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=jingyaogong/minimind-v&type=Date&theme=dark"/>
</picture>

# 🎓 引用

如果您觉得 MiniMind-V 对您的研究或工作有所帮助，请引用：

```bibtex
@misc{minimind-v,
  title = {MiniMind-V: Train a Tiny VLM from Scratch},
  author = {Jingyao Gong},
  year = {2024},
  url = {https://github.com/jingyaogong/minimind-v},
  note = {GitHub repository, accessed 2026}
}
```

# 📜 许可协议

本仓库遵循 [Apache-2.0 License](LICENSE) 开源协议。
