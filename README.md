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

* 此项目旨在从0开始，仅用1.3块钱成本 + 1小时！即可训练出67M参数的超小多模态视觉语言模型**MiniMind-V**。
* **MiniMind-V**最小版本体积仅为 GPT3 的约 $\frac{1}{2600}$，力求做到个人GPU也可快速推理甚至训练。
* **MiniMind-V**是[MiniMind](https://github.com/jingyaogong/minimind)纯语言模型的视觉能力额外拓展。
* 项目同时包含了VLM大模型的极简结构、数据集清洗、预训练(Pretrain)、监督微调(SFT)等全过程代码。
* 这不仅是一个开源VLM模型的最小实现，也是入门视觉语言模型的简明教程。
* 希望此项目能为所有人提供一个抛砖引玉的示例，一起感受创造的乐趣！推动更广泛AI社区的进步！

> 为防止误解，“1小时” 基于NVIDIA 3090硬件设备（单卡）测试`1 epoch`，“1.3块钱” 指GPU服务器租用成本。



<div align="center">

![minimind-3v](./images/minimind-3v.gif)

[🔗🤖在线体验](https://www.modelscope.cn/studios/gongjy/MiniMind-V) | [🔗🎞️视频介绍](https://www.bilibili.com/video/BV1Sh1vYBEzY)

</div>

# 📌 项目介绍

“用乐高拼出一架飞机，远比坐在头等舱里飞行更让人兴奋！”
构建VLM范式的多模态大模型是否真的如想象中那样复杂？它的代码实现到底如何？
训练过程究竟难不难？那么现在，探索它们的答案，一起感受创造的乐趣吧！

> [!TIP]
> （截至2026-02-15）MiniMind-V 系列已完成了以下型号模型训练，最小仅需67M (0.067B)，即可具备识图和对话的能力！

| 模型 (大小)                   | 推理占用   | release    | 
|---------------------------|--------|------------|
| minimind-3v-moe (201M-A67M) | 1.0 GB | 2026.04.01 |
| minimind-3v (67M)         | 0.5 GB | 2026.04.01 |
| MiniMind2-V (104M)        | 1.1 GB | 2025.02.20 |
| MiniMind2-Small-V (26M)   | 0.6 GB | 2025.02.20 |
| minimind-v-v1-small (27M) | 0.6 GB | 2024.10.04 |
| minimind-v-v1 (109M)      | 1.1 GB | 2024.10.04 |

#### 👉 更新日志

<details> 
<summary> <b>2026-04-01</b> </summary>

- 新增 minimind-3v (67M) 和 minimind-3v-moe (201M-A67M) 模型
- 统一使用768+8架构，支持dense和moe两种模式
- 视觉编码器从CLIP切换为SigLIP2（siglip2-base-p16-ve）
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
<summary> <b>2025-04-27</b> </summary>

- 兼容性更新
- 适配[「minimind仓库新特性」](https://github.com/jingyaogong/minimind/issues/370)
- 规范化部分代码

</details>

<details>

<summary> <b>More...</b> </summary>

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
* Python==3.10.16
* [requirements.txt](./requirements.txt)

</details>

### 第0步

```bash
# 克隆代码仓库
git clone --depth 1 https://github.com/jingyaogong/minimind-v
```

```bash
# 下载siglip2模型到 ./model 目录下
git clone https://huggingface.co/jingyaogong/siglip2-base-p16-ve
# 或
git clone https://modelscope.cn/models/gongjy/siglip2-base-p16-ve
```

```bash
# 下载minimind语言模型权重到 ./out 目录下（作为训练VLM的基座语言模型）
# HuggingFace
https://huggingface.co/jingyaogong/minimind-3v-pytorch/blob/main/llm_768.pth
# 国内源
https://modelscope.cn/models/gongjy/minimind-3v-pytorch/resolve/master/llm_768.pth
```

## Ⅰ 测试已有模型效果

### 1' 环境准备

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2' 下载模型

```bash
git clone https://huggingface.co/jingyaogong/minimind-3v
```

### 3' 命令行问答

```bash
# load_from='model': 加载原生PyTorch权重, load_from='其他路径': 加载transformers格式
python eval_vlm.py --load_from model --weight sft_vlm

# 或使用transformers格式模型
python eval_vlm.py --load_from minimind-3v
```

### 4' 启动WebUI（可选）

```bash
# ⚠️ 须先将 transformers 格式模型文件夹复制到 ./scripts/ 目录下（例如：cp -r minimind-3v ./scripts/minimind-3v），web_demo_vlm 脚本会自动扫描该目录下包含权重文件的子文件夹，如不存在则报错
cd scripts && python web_demo_vlm.py
```

## Ⅱ 从0开始自己训练

### 1' 环境准备

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

<details>
<summary>注：提前测试Torch是否可用cuda</summary>

```bash
import torch
print(torch.cuda.is_available())
```

如果不可用，请自行去[torch_stable](https://download.pytorch.org/whl/torch_stable.html)
下载whl文件安装。参考[链接](https://blog.csdn.net/weixin_45456738/article/details/141029610?ops_request_misc=&request_id=&biz_id=102&utm_term=%E5%AE%89%E8%A3%85torch&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-141029610.nonecase&spm=1018.2226.3001.4187)

</details>

### 2' 数据下载

从下文提供的[数据集链接](https://huggingface.co/datasets/jingyaogong/minimind-v_dataset)
下载所需内容并放到`./dataset`下。

<details>
<summary>注：数据集须知</summary>

【注1】之前需解压50万零碎的图像文件可能非常慢。2025-12-27起，数据集格式统一为Parquet，图文一体化存储，体积更小，无需解压，加载更快。

【注2】Parquet是列式存储格式，支持高效压缩和快速读取。如果你对它感到陌生，可以预览数据内容，在`dataset/`目录下执行`python lm_dataset.py`可视化前5条图文对

Pretrain数据：
```bash
wget https://hf-mirror.com/datasets/jingyaogong/minimind-v_dataset/resolve/main/pretrain_i2t.parquet
```

SFT数据：
```bash
wget https://hf-mirror.com/datasets/jingyaogong/minimind-v_dataset/resolve/main/sft_i2t.parquet
```

建议预留~2GB空间存放数据集，若无多余空间存放pretrain数据，可尝试跳过pretrain训练步骤直接进行sft训练。

</details>

### 3' 开始训练

**3.1 预训练（学图像描述）**

```bash
# 基础训练命令（从LLM权重开始，仅训练vision_proj）
python train_pretrain_vlm.py --epochs 4 --from_weight llm
```

> 执行预训练，得到 `pretrain_vlm_*.pth` 作为预训练的输出权重（其中*为模型的dimension，默认为768）


**3.2 监督微调（学看图对话方式）**

```bash
# 基础训练命令（从预训练权重开始，全参数微调）
python train_sft_vlm.py --epochs 2 --from_weight pretrain_vlm
```

> 执行监督微调，得到 `sft_vlm_*.pth` 作为指令微调的输出权重

<details>
<summary>注：训练须知</summary>

**训练特性：**
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
- `--freeze_llm`: 是否冻结LLM参数（仅pretrain使用）
- 更多可直接参考代码

</details>


---

### 4' 测试模型效果

确保需要测试的模型`*.pth`文件位于`./out/`目录下。
也可以直接去[此处](https://huggingface.co/jingyaogong/minimind-3v-pytorch)下载使用我训练的`*.pth`文件。

```bash
# 测试SFT模型（默认）
python eval_vlm.py --weight sft_vlm

# 测试Pretrain模型
python eval_vlm.py --weight pretrain_vlm
```

---

> [!TIP]
> 训练脚本均为Pytorch原生框架，均支持多卡加速，假设你的设备有N (N＞1) 张显卡：

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

MiniMind-V (VLM)的基座语言模型MiniMind (LLM)来自孪生项目[minimind](https://github.com/jingyaogong/minimind)，
具体的模型结构、训练细节、原理、测试效果等均可移步[minimind](https://github.com/jingyaogong/minimind)项目查阅。
此处为减少冗余，省略讨论LLM的相关部分，默认您已对MiniMind (LLM)的细节有基本的了解。

> 即使您不太了解LLM的细节，也可参考“快速开始”流程训练一个MiniMind-V，
> 这并不受到影响，仓库致力于最低成本的开箱即用！

MiniMind-V的结构仅增加Visual Encoder和特征投影两个子模块，增加模态混合分支，以支持多种模态信息的输入：
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
具体使用[siglip2-base-p16-ve](https://huggingface.co/jingyaogong/siglip2-base-p16-ve)，
一种基于 ViT-B/16 架构的Visual Encoder用于描述图像文本信息。
当前使用的 SigLIP2 NaFlex 视觉编码器会根据预处理结果生成最多256个patch token作为encoder编码层的输入，
最终产生1×768维的嵌入向量用于和文本对计算误差。
我们并不需要最终嵌入表示，因此只取encoder层的输出，也就是VIT核心主干的输出特征即可。
它拿到前一层256×768大小的特征，通过reshape将每4个相邻token拼接为1个（256×768 → 64×3072），再经过2层MLP（Linear→GELU→Linear）投影到LLM的隐藏维度，最终作为64个visual token输入MiniMind-V。
与LLM的结合在获取图像encoder特征后，一方面需要把视觉特征对齐到LLM的文本token维度，
另一方面，要将图像特征映射到与文本embedding相同的空间，即文本token和原生的视觉token需要磨合并不能直接地一视同仁，
可以称之为跨模态的特征对齐。

[LlaVA-1](https://arxiv.org/pdf/2304.08485)使用简单的线性变换完成对齐，[LlaVA-1.5](https://arxiv.org/pdf/2310.03744)升级为2层MLP，MiniMind-V采用与LlaVA-1.5相同的MLP Projection方案，并结合reshape进行token压缩。

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

在`minimind-v`中，使用64个`<|image_pad|>`组成的占位符代替图像（SigLIP2输出的256个patch特征经reshape+MLP压缩为64个token），因此`minimind-v`的prompt为：

```text
<|image_pad|><|image_pad|>...<|image_pad|>(×64)\n这个图片描述的是什么内容？
```

计算完embedding和projection，用视觉特征替换掉对应占位符的embedding后，整个计算过程到输出则和LLM部分没有差异。

![input](./images/minimind-v-input.jpg)

至此，`MiniMind-V`的所有细节呈现完毕，VLM模型子类继承自`MiniMind`，仅做**最小**变更而产生，核心算法改动`< 50行`，迁移难度极低，和`LlaVA`等模型具体实现存在区别，但思路一致。

# 📌 实验

## Ⅰ 数据集

原始来源：
- [Chinese-LLaVA-Vision](https://huggingface.co/datasets/LinkSoul/Chinese-LLaVA-Vision-Instructions)：包含约57万张预训练图像，来自CC-3M和COCO 2014
- [llava-en-zh-300k](https://huggingface.co/datasets/BUAADreamer/llava-en-zh-300k)：包含300k条指令微调数据和15万张图像
- [LLaVA-SFT-665K](https://huggingface.co/datasets/csuhan/LLaVA-SFT-665K)：包含665k条指令微调数据

其中部分为中文数据，部分为英文数据。问答内容经过翻译，对中文支持更友好，进一步经过整理并`resize`（pretrain分辨率128×128，sft分辨率160×160）。

(pretrain_i2t.parquet) 预训练数据集格式：

```text
列名: conversations (json string), image_bytes (binary), image_names (string)

conversations 示例:
[
  {"role": "user", "content": "提供给定图像的简要描述。\n<image>"},
  {"role": "assistant", "content": "橄榄油是自由使用的健康成分。"}
]
image_bytes: <图像二进制数据>
```

(sft_i2t.parquet) 单图指令微调数据集格式：

```text
列名: conversations (json string), image_bytes (binary), image_names (string)

conversations 示例:
[
  {"role": "user", "content": "闹钟的位置对睡眠质量有什么影响？<image>"},
  {"role": "assistant", "content": "把数字闹钟放在床头柜..."}
]
image_bytes: <图像二进制数据>
```

> 注：sft_i2t.parquet 共约 58 万条数据，其中约 23.6 万条为含图对话（i2t），约 34.6 万条为纯文本对话（t2t），后者用于保持模型的基础语言能力。

数据集下载地址：([ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind-v_dataset) | [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind-v_dataset))

## Ⅱ 训练

训练分为两个阶段，均冻结Visual Encoder梯度，仅训练Projection和LLM部分。
训练基于LLM预训练权重初始化，支持DDP多卡训练、混合精度（bfloat16）、torch.compile加速和swanlab日志记录。

> train_pretrain_vlm

预训练阶段从约113万条图文描述数据中学习图片的通用知识（如鹿是鹿，狗是狗）。
此阶段采用较高学习率（1e-4），最大序列长度360，冻结LLM主体参数，仅设置Projection和LLM的第0层可学习，
目的是让模型快速建立视觉特征到语言空间的基础映射，同时避免破坏LLM已有的语言能力。

> train_sft_vlm

指令微调阶段从约58万条数据中学习真实问答格式，其中约23.6万条为图文多轮对话，约34.6万条为纯文本对话（用于保持LLM基础能力）。
此阶段采用较低学习率（1e-5~1e-6），最大序列长度768，解冻Projection和LLM全部参数进行全量微调，
使模型学会根据图片内容进行多轮对话，并通过混入的纯文本数据缓解灾难性遗忘。

> 训练时间和Loss走势（仅供参考）

Pretrain [768+8] (dense & moe)
![input](./images/pretrain_loss.jpg)

SFT [768+8] (dense & moe)
![input](./images/sft_loss.jpg)

## Ⅲ 模型权重

| 模型格式 | ModelScope | HuggingFace |
|---|---|---|
| 原生PyTorch (`*.pth`) | [minimind-3v-pytorch](https://www.modelscope.cn/models/gongjy/minimind-3v-pytorch) | [minimind-3v-pytorch](https://huggingface.co/jingyaogong/minimind-3v-pytorch) |
| Transformers 格式 | [minimind-v collection](https://modelscope.cn/collections/MiniMind-V-42b841dde22d41) | [minimind-v collection](https://huggingface.co/collections/jingyaogong/minimind-v-67000833fb60b3a2e1f3597d) |

> 注：Transformers版本均为单图指令微调后的`MiniMind-V`模型

# 📌 评估

### 效果测试

#### 单图对话

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
      <td>在这幅图片中，一架白色的飞机正降落在一片广阔的天空中。天空中飘浮着许多流线型的建筑物，这些建筑物散落在天空中。除了主要的飞机外，还有两辆汽车在场景中各处停放，包括一辆巴士和一辆小型汽车。这架飞机似乎停在地面上，表明它们正在进行商业活动。</td>
      <td>在这张照片中，有一架大型的飞机正在降落，这表明它是为这架飞机而设计的。此外，它停在云层之下，这表明它在移动。天空中有云朵，暗示着这架飞机正在空中飞行。整个场景的背景显示出一种宁静祥和的气氛，暗示这架飞机正在飞行中作为一个机会来进行外交活动或与其他客机接触。</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/birthday-cake-candles-table.jpg" alt="birthday-cake">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>图片显示了一张大生日蛋糕，上面刻有各种生日和装饰物的生日和一束蜡烛。这些甜蜜而独特的气味使其成为这个特殊场合的焦点。蛋糕被放在餐桌上，有蛋糕放在上面，很可能是放在桌上，用蜡烛加上蜡烛来营造出诱人的场景。此外，这张生日蛋糕还配上了蜡烛作为装饰物，进一步丰富了整个生日的氛围。</td>
      <td>这张照片捕捉到了一个生日聚会，各种美味可口的生日蛋糕和一把勺子放在那里。它被放置在一个充满生气的蛋糕盒子里，里面装着各种各样的草莓糖霜糖浆。蛋糕上有一个大号的蜡烛，给人一种温馨和诱人的图案。周围有很多小点心，比如小心地蜡烛和糖霜，让整个蛋糕看起来更加令人放松和诱人。</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/pizza-on-wooden-board.jpg" alt="pizza">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>在这张图片中，有一个装饰着奶酪的比萨饼和一片新鲜的青绿野餐毯。这看起来像是一块沙拉，给人一种清新、诱人、美味的享受体验。披萨的大小和大小暗示着一种随意和随意的用餐体验。画面中，一群人围坐在一块砖块上，其中一些则分散在桌子上，周围摆放着不同种类的青绿野餐毯。此外，桌子上还放着一块披萨片，上面摆放着各种各样的青绿野餐毯，营造出一种轻松愉快的氛围。</td>
      <td>图中，比萨饼在木桌上。它被切成了比萨饼的大小，而且披萨放置在木桌上。比萨有很多配料，包括奶酪和各种酱汁。披萨有多种口味，包括经典的牛排式和意大利式，还有一些加在比萨表面。比萨上有很多新鲜水果，如西红柿和莫吉托，以及奶酪，使这幅图片更加丰富和有吸引力。</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/red-sports-car-road.jpg" alt="red-car">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>在这幅图片中，一辆白色的马车停泊在一条红色的路牌上。这个车辆位于一条绿油油的道路上，很可能是一个购物中心或高速公路。在这辆车的后部，可以看到一个绿色的马车停泊在路上，这是典型的户外场所。这辆马车可能是为了娱乐或观赏车辆而停放。</td>
      <td>画面显示了一辆红色高性能赛车，停在一辆大型汽车后面。这辆车可以看到车身、汽车、汽车及汽车停在里面。它似乎是一辆大型红色汽车，有各种大小的汽车，表明它可能是汽车制造商生产的。此外，车辆周围的环境暗示了一种户外环境，因为一辆汽车也出现在场景中。</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/row-of-colorful-houses.jpg" alt="colorful-houses">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>画面中，一座蓝色白色的大房子位于一条城市街道上，为这个地区增添了一丝自然与奇思妙想。墙上挂着一盏交通灯，为整个场景增添了特色和引人入胜。</td>
      <td>画面中，一座蓝色房子旁边有很多小花瓶。这表明这座房子里可能正在营运一些小型小型花盆或盆栽植物。一些人聚集在房子周围，可能正在享受户外生活中的某种乐趣或美景。有些人站在画面中，而其他人则散布于场景中。总体而言，这幅场景捕捉到了一个美丽而令人愉快的场景，展示出该房子里一个令人放松和宁静的环境。</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/snow-mountain-lake-view.jpg" alt="snow-mountain">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>在这幅图片中，有一个高高的山，它看起来像是大森林中的一片高山。天空中闪烁着不同颜色的星星，给画面增加了一抹红点。天空中有两朵高大的树，树木高高地挂在一起，暗示着森林中的宁静与自然之美。在画面的中心，可以看到一棵高大的松树，树干上覆盖着一层薄薄的苔藓。这种高高的松树与周围的大山构成了一个有趣而引人入胜的背景，为这片自然之美增添了一丝神秘色彩。</td>
      <td>这幅图片展示了一个令人印象深刻的宁静湖面。湖水从天上飘浮着，暗示着一个令人平静和放松的水面。湖边上有几匹高大、形状各异的景象，它们在湖面上显得格外美丽。此外，在湖的边缘，有一座巨大的高山，为整个湖景增添了几分神秘色彩。湖景中的天空也被描述得如画，给整个画面增添了一种纯净和宁静的气氛。</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/street-food-hotpot-table.jpg" alt="street-food">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>图中，一大群人聚集在一张大餐桌旁欣赏着烤肉和热带水果，其中一人站在一碗里摆放着各种各样的盘子。桌子上放着几个碗，上面摆满了肉类和其他配料。有些放在盘子里，其他的则放在桌子上。在这张餐桌周围，有几个盘子，其中两个放着一杯酒，另一个放在靠近餐桌的左侧。</td>
      <td>画面中，一群人聚集在一家大餐馆里，享受着一顿饭。这家餐厅的菜单上有一些生菜和猪肉，但它们已经被切成了四份，上面还有一个碗。他们拿着烤肉准备食用。在背景中，有几个瓶子在场景中。还有一把勺子位于桌子左侧，使盘子看起来更吸引人。一盆盆栽植物放在桌子左侧上方，为空间增添了一抹绿色。场景中的其他元素包括一个碗，里面放着葡萄酒和两根葡萄酒。</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/three-kittens-basket.jpg" alt="kittens">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>图中，一只棕色的小灰猫正坐在篮子里。这只猫身上戴着一顶棕色的帽子，很可能是一个戴帽子的男人。在篮子里，一只棕色的紫色小灰猫正沿着篮子里去休息。这些小猫似乎也在享受这份温暖，但它们似乎并没有完全放过来。此外，背景中还可以看到一把剪刀。这把剪刀看起来是专门为小猫设计的，它可以用作家庭相册或礼物。在篮子的侧边，有几条篮子，其中一条是最亮的，另一条则是最暗的。在篮子中可以看到一只棕色小灰猫，而另一条则是更暗的。</td>
      <td>在这张照片中，一只小猫坐在篮子里，紧挨着它坐在篮子里的那块木篮上。猫的身体上有九条纹毛发。这个场景描绘了它们之间的亲密关系，展示了它们在一起度过时光的不同场合。画面中，一群大的猫坐在篮子里，其中一只小猫也被描述为小猫，这可能表明他们正在享受与猫互动、与它们的互动或一起度过愉快时光。</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/tropical-beach-palm-tree.jpg" alt="tropical-beach">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>在这张图片中，沙滩上有很多椅子，还有一些人站着，可以看到一把遮阳伞。虽然它看起来很大，但却没有任何特别的设计。沙滩上有许多椅子，表明这是一家餐馆或者服务员办公室。其中最引人注目的是一张海边椅子，椅子上放着一只热带海滩椅。这个椅子非常适合放松身心、享受海滩时光。此外，还有一些椅子和其他人在场景中，可能是为了放置食物或其他用途。靠近椅子的椅子表示该位置可供使用的其他人使用，也许也有一人在靠近那个椅子的地方。</td>
      <td>图片显示了一个美丽的海滩场景，有很多椅子散布在天然的棕榈树上。其中一个椅子靠近海滩，而另一个则较小。沙滩上有两把椅子，其中一把靠近中间，另一把则稍微偏左，还有一些则在靠近边缘处。在海边的海滩周围，你可以看到几个人坐在海边的沙滩上，有的靠近海水中，还有一张沙滩椅。其中一张椅子靠近海滩，另一张椅子靠近海边。此外，还可以看到几只遮阳伞，为沙滩上的躺椅提供了遮阴。</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/yellow-school-bus-road.jpg" alt="school-bus">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>画面中，一辆蓝色的黄色公共汽车正从一辆黄色公共汽车驶过，在道路上停泊着。这辆公共汽车看起来是在一个黄色的黄色高速公路上。图中有几个人，其中一些靠近前景，而另一些则靠后一些，但都没有看到。在黄色公共汽车附近，可以看到一辆停在路边，那辆停在路边。此外，还有两辆不同方向的巴士，一辆靠近前景，另一辆靠近前景，另一辆稍微靠后一些。</td>
      <td>画面中，一辆黄色和黄色相间的黄色和蓝色交叉路口的蓝色公共汽车正在一条通往路缘上的红色公交车站。有几辆公共汽车正停在路边，它们离一排车道很近。在背景中，可以看到一些长凳，它们在城市里交叉起来。一个长凳位于图中最左侧，而另一个则稍微靠后一点，为画面增添了一些城市特色。整个场景中有很多人和车辆散落在场景各处，包括黄色和蓝色的交叉路口。整个场景给人一种忙碌和迷茫的感觉，这也突显了公共汽车在市区中的存在和目的。</td>
    </tr>
  </tbody>
</table>

### 效果小结：

两个模型均能识别图像主体（飞机、蛋糕、汽车、海滩等），但普遍存在重复表述和幻觉细节。受限于模型和数据规模，整体处于"能看懂大意、细节不准"的阶段。

视觉信号对于LLM视作一种特殊的外语，因此"学习外语"的能力高低，很大程度上取决于LLM的能力。LLM性能越强，对应的VLM越强，此时效果增益会很明显。

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
