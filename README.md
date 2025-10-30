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

* 此项目旨在从0开始，仅用1.3块钱成本 + 1小时！即可训练出26M参数的超小多模态视觉语言模型**MiniMind-V**。
* **MiniMind-V**最小版本体积仅为 GPT3 的约 $\frac{1}{7000}$，力求做到个人GPU也可快速推理甚至训练。
* **MiniMind-V**是[MiniMind](https://github.com/jingyaogong/minimind)纯语言模型的视觉能力额外拓展。
* 项目同时包含了VLM大模型的极简结构、数据集清洗、预训练(Pretrain)、监督微调(SFT)等全过程代码。
* 这不仅是一个开源VLM模型的最小实现，也是入门视觉语言模型的简明教程。
* 希望此项目能为所有人提供一个抛砖引玉的示例，一起感受创造的乐趣！推动更广泛AI社区的进步！

> 为防止误解，“1小时” 基于NVIDIA 3090硬件设备（单卡）测试`1 epoch`，“1.3块钱” 指GPU服务器租用成本。



<div align="center">

![minimind2-v](./images/minimind2-v.gif)

[🔗🤖在线体验](https://www.modelscope.cn/studios/gongjy/MiniMind-V) | [🔗🎞️视频介绍](https://www.bilibili.com/video/BV1Sh1vYBEzY)

</div>

# 📌 Introduction

“用乐高拼出一架飞机，远比坐在头等舱里飞行更让人兴奋！”
构建VLM范式的多模态大模型是否真的如想象中那样复杂？它的代码实现到底如何？
训练过程究竟难不难？那么现在，探索它们的答案，一起感受创造的乐趣吧！

> [!TIP]
> （截至2025-02-20）MiniMind-V 系列已完成了以下型号模型训练，最小仅需26M (0.026B)，即可具备识图和对话的能力！

| 模型 (大小)                   | 推理占用   | release    | 
|---------------------------|--------|------------|
| MiniMind2-V (104M)        | 0.6 GB | 2025.02.20 |
| MiniMind2-Small-V (26M)   | 1.1 GB | 2025.02.20 |
| minimind-v-v1-small (27M) | 0.6 GB | 2024.10.04 |
| minimind-v-v1 (109M)      | 1.1 GB | 2024.10.04 |

### 👉**最近更新**

<details close> 
<summary> <b>2025-10-24</b> </summary>

- bug修复：模型权重不对应
- 适配[「minimind-1024更新」](https://github.com/jingyaogong/minimind)
- 代码重构：训练和评估脚本规范化
- 新增完整的断点续训支持

</details>

<details close> 
<summary> <b>2025-04-27</b> </summary>

- 兼容性更新
- 适配[「minimind仓库新特性」](https://github.com/jingyaogong/minimind/issues/370)
- 规范化部分代码

</details>

<details close> 
<summary> <b>2025-02-20</b> </summary>

- MiniMind2-V伴随MiniMind2同步更新
- 大幅减少所有冗余代码，规范代码格式
- 大幅精简模型冗余结构
- 更新数据集格式，拓展新的SFT数据集
- 比前代VLM更优秀的效果！

</details>

<details close>

<summary> <b>More...</b> </summary>

**2024-10-05**

- MiniMind-V如期而至，首次开源

</details>

# 📌 快速开始

<details style="color:rgb(128,128,128)">
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
git clone https://github.com/jingyaogong/minimind-v
```

```bash
# 下载clip模型到 ./model/vision_model 目录下
git clone https://huggingface.co/openai/clip-vit-base-patch16
# or
git clone https://www.modelscope.cn/models/openai-mirror/clip-vit-base-patch16
```

```bash
# 下载minimind语言模型权重到 ./out 目录下（作为训练VLM的基座语言模型）
# HuggingFace
https://huggingface.co/jingyaogong/MiniMind2-V-PyTorch/blob/main/llm_512.pth # or llm_768.pth
# 国内源
https://modelscope.cn/models/gongjy/MiniMind2-V-PyTorch/resolve/master/llm_512.pth # or llm_768.pth
```

## Ⅰ 测试已有模型效果

### 1.环境准备

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2.下载模型

```bash
git clone https://huggingface.co/jingyaogong/MiniMind2-V
```

### 3.命令行问答

```bash
# load_from='model': 加载原生PyTorch权重, load_from='其他路径': 加载transformers格式
python eval_vlm.py --load_from model --weight sft_vlm

# 或使用transformers格式模型
python eval_vlm.py --load_from MiniMind2-V
```

### 4.或启动WebUI

```bash
python web_demo_vlm.py
```

## Ⅱ 从0开始自己训练

### 1.环境准备

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

<details style="color:rgb(128,128,128)">
<summary>注：提前测试Torch是否可用cuda</summary>

```bash
import torch
print(torch.cuda.is_available())
```

如果不可用，请自行去[torch_stable](https://download.pytorch.org/whl/torch_stable.html)
下载whl文件安装。参考[链接](https://blog.csdn.net/weixin_45456738/article/details/141029610?ops_request_misc=&request_id=&biz_id=102&utm_term=%E5%AE%89%E8%A3%85torch&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-141029610.nonecase&spm=1018.2226.3001.4187)

</details>

### 2.数据下载

从下文提供的[数据集链接](https://huggingface.co/datasets/jingyaogong/minimind-v_dataset)
下载所需内容并放到`./dataset`下。

<details style="color:rgb(128,128,128)">
<summary>注：数据集须知</summary>

Pretrain数据：
```bash
wget https://hf-mirror.com/datasets/jingyaogong/minimind-v_dataset/resolve/main/pretrain_data.jsonl
wget https://hf-mirror.com/datasets/jingyaogong/minimind-v_dataset/resolve/main/pretrain_images.zip
unzip pretrain_images.zip && rm pretrain_images.zip
```

SFT数据：
```bash
wget https://hf-mirror.com/datasets/jingyaogong/minimind-v_dataset/resolve/main/sft_data.jsonl
wget https://hf-mirror.com/datasets/jingyaogong/minimind-v_dataset/resolve/main/sft_images.zip
unzip sft_images.zip && rm sft_images.zip
```

`*.jsonl`为问答文本，`*images`为配套的图片数据，下载完成后需要解压图像数据。

请预留~5GB空间存放数据集，若无多余空间存放pretrain数据，可尝试跳过pretrain训练步骤直接进行sft训练。

</details>

### 3.开始训练

**3.1 预训练（学图像描述）**

```bash
# 基础训练命令（从LLM权重开始，仅训练vision_proj）
python trainer/train_pretrain_vlm.py --epochs 4 --from_weight llm
```

> 执行预训练，得到 `pretrain_vlm_*.pth` 作为预训练的输出权重（其中*为模型的dimension，默认为512）


**3.2 监督微调（学看图对话方式）**

```bash
# 基础训练命令（从预训练权重开始，全参数微调）
python trainer/train_sft_vlm.py --epochs 2 --from_weight pretrain_vlm
```

> 执行监督微调，得到 `sft_vlm_*.pth` 作为指令微调的输出权重

<details style="color:rgb(128,128,128)">
<summary>注：训练须知</summary>

**训练特性：**
- 支持断点续训：添加`--from_resume 1`参数可从上次中断处继续训练
- 支持GPU数量变化：续训时GPU数量改变会自动转换step
- 原子性保存：使用临时文件+替换机制，防止保存过程中断导致权重损坏
- 每次保存同时生成`out/**.pth`（模型权重）和`checkpoints/**_resume.pth`（训练状态）文件

```bash
# 训练中断后，使用相同命令并添加 --from_resume 1
python trainer/train_sft_vlm.py --epochs 4 --from_resume 1
```

**参数说明：**
- `--from_weight`: 基础权重名称（llm, pretrain_vlm, none等）
- `--save_weight`: 保存权重的前缀名
- `--from_resume`: 是否续训（0=从头开始，1=从检查点继续）
- `--freeze_llm`: 是否冻结LLM参数（仅pretrain使用）
- 更多可直接参考代码

</details>


---

### 4.测试模型效果

确保需要测试的模型`*.pth`文件位于`./out/`目录下。
也可以直接去[此处](https://huggingface.co/jingyaogong/MiniMind2-V-PyTorch)下载使用我训练的`*.pth`文件。

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

<details style="color:rgb(128,128,128)">
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

# 📌 VLM Detail

MiniMind-V (VLM)的基座语言模型MiniMind (LLM)来自孪生项目[minimind](https://github.com/jingyaogong/minimind)，
具体的模型结构、训练细节、原理、测试效果等均可移步[minimind](https://github.com/jingyaogong/minimind)项目查阅。
此处为减少冗余，省略讨论LLM的相关部分，默认您已对MiniMind (LLM)的细节有基本的了解。

> 即使您不太了解LLM的细节，也可参考“快速开始”流程训练一个MiniMind-V，
> 这并不受到影响，仓库致力于最低成本的开箱即用！

MiniMind-V的结构仅增加Visual Encoder和特征投影两个子模块，增加模态混合分支，以支持多种模态信息的输入：
![LLM-structure](./images/VLM-structure.png)
![LLM-structure](./images/VLM-structure-moe.png)


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
和LlaVA、Qwen-VL等视觉语言模型类似，MiniMind-V同样选用开源Clip系列模型作为Visual Encoder。
具体使用[clip-vit-base-patch16](https://huggingface.co/openai/clip-vit-base-patch16)，
一种基于 ViT-B/16 架构的经典Visual Encoder用于描述图像文本信息。
输入的图像尺寸为224x224，因为划分的Patch是16×16，所以会产生14*14=196个token作为encoder编码层的输入，
最终产生1×768维的嵌入向量用于和文本对计算误差。
我们并不需要最终嵌入表示，因此只取encoder层的输出，也就是VIT核心主干的输出特征即可。
它拿到前一层维度196×768大小的特征，我们把它作为196个visual token输入MiniMind-V。
与LLM的结合在获取图像encoder特征后，一方面需要把768维度的visual token对齐到LLM的文本token，
另一方面，要将图像特征映射到与文本embedding相同的空间，即文本token和原生的视觉token需要磨合并不能直接地一视同仁，
可以称之为跨模态的特征对齐。
[LlaVA-1](https://arxiv.org/pdf/2304.08485)使用简单的无偏线性变换完成了这一操作，效果很不错，MiniMind-V同样如此。

![llava-structure](./images/llava-structure.png)

至此，MiniMind-V的内部结构变化已经呈现完毕。

</details>


---

下面，我们简单讨论MiniMind-V的外部输入输出的变化。

VLM的输入依然是一段文本，其中包含特殊的`<image>`占位符。
在计算文本嵌入后，可以将图像编码器生成的向量投影到该占位符对应的嵌入部分，替换掉原先的占位符embedding。
例如：

```text
<image>\n这个图像中有什么内容？
```

在`minimind-v`中，使用196个字符组成的 `@@@...@@@`
占位符代替图像，之所以是196个字符，前面有所提及：
任何图像都被clip模型encoder为196×768维的token，
因此`minimind-v`的prompt为：

```text
@@@......@@@\n这个图片描述的是什么内容？
```

计算完embedding和projection，并对图像部分token替换后整个计算过程到输出则和LLM部分没有任何区别。

![input](./images/minimind-v-input.png)

一次性多图的实现方法就是通过注入多个`<image>`图像占位符进行实现，不需要修改任何框架。

<details>
<summary> 视频理解的拓展思路 </summary>

write by [@xinyanghuang7](https://github.com/xinyanghuang7)

对于多模态大模型的视频理解能力，一个可行的思路是参考现有MiniCPM-V 2.6 进行视频理解的Python示例。
主要思想是通过提取视频关键帧，而后进行多图推理。
因此，如果希望在MiniMind-V中添加视频理解能力，可以在现有多图训练的基础上，参考此python脚本中对于关键帧的提取方法，而后加大训练文件中支持图片的数量。
所支持的MAX_NUM_FRAMES越多，所消耗的显存越大。

```text
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu  # pip install decord

model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
                                  attn_implementation='sdpa',
                                  torch_dtype=torch.bfloat16)  # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

MAX_NUM_FRAMES = 64  # if cuda OOM set a smaller number


def encode_video(video_path):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames


video_path = "video_test.mp4"
frames = encode_video(video_path)
question = "Describe the video"
msgs = [
    {'role': 'user', 'content': frames + [question]},
]

# Set decode params for video
params = {}
params["use_image_id"] = False
params["max_slice_nums"] = 2  # 如果cuda OOM且视频分辨率大于448*448可设为1

answer = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer,
    **params
)
print(answer)
```

</details>

至此，`MiniMind-V`的所有细节已经呈现完毕。
`MiniMind-V`的模型子类完全继承自`MiniMind`，
仅基于后者做**最小**变更而产生，
其核心算法改动`< 50行`，迁移难度极低。
因此可能和`LlAVA`等模型细节可能存在区别，但思路完全统一。

# 📌 Experiment

## Ⅰ 数据集

来源：[Chinese-LLaVA-Vision](https://huggingface.co/datasets/LinkSoul/Chinese-LLaVA-Vision-Instructions)
包含约57万张预训练图像，来自CC-3M和COCO 2014；
[llava-en-zh-300k](https://huggingface.co/datasets/BUAADreamer/llava-en-zh-300k)
包含300k条指令微调数据和15万张图像。
问答内容经过翻译，
对中文支持更友好，进一步经过整理并`resize`。

(pretrain_vlm_data.jsonl) 预训练数据集格式：

```json lines
{
  "conversations": [
    {
      "role": "user",
      "content": "提供给定图像的简要描述。\n<image>"
    },
    {
      "role": "assistant",
      "content": "橄榄油是自由使用的健康成分。"
    }
  ],
  "image": "GCC_train_002582585.jpg"
}
```

(sft_vlm_data.jsonl) 单图指令微调数据集格式：

```json lines
{
  "conversations": [
    {
      "role": "user",
      "content": "闹钟的位置对睡眠质量有什么影响？<image>"
    },
    {
      "role": "assistant",
      "content": "把数字闹钟放在床头柜..."
    }
  ],
  "image": "train-00000-of-00001_image_0_0.jpg"
}
```

(sft_vlm_data_multi.jsonl) 多图指令微调数据集格式：

```json lines
{
  "conversations": [
    {
      "role": "user",
      "content": "context: Source Image: <image> Target Image: <image> Instruction: What is the correct image edit instruction that can transfrom the source image to target image?<image>"
    },
    {
      "role": "assistant",
      "content": "take the people out of the back in the photo. Remove the two people behind the woman in the white dress and the man in the blue suit. remove people behind the couple in the centre"
    }
  ],
  "image": "0.jpg, 1.jpg"
}
```

<details>
<summary> 数据说明 </summary>

* 多图数据集规模相对较小且为英文对话，数据集仅包含两图对比的场景，因此微调效果有限，这里只提供一种参考思路。


* `jsonl`均为文本指令，`images.zip`均为配套的图像数据（下载后需要解压）

</details>

数据集下载地址：([ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind-v_dataset) | [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind-v_dataset))

## Ⅱ 训练

> train_pretrain_vlm

预训练从595K条数据集中学习图片的通用知识，比如鹿是鹿，狗是狗。

> train_sft_vlm

指令微调从300K条真实对话数据集中学习对图片提问的真实问答格式，更符合与人类的交流习惯。

> train_sft_vlm

多图微调提供demo：鸟类对比数据集，长度为13.6k的真实问答格式。

训练时均冻结visual encoder也就是clip模型梯度，
只训练Projection和LLM两部分。
预训练中，只设置Projection和LLM的最后一层参数可学习。
指令微调中，设置Projection和LLM的全部参数可学习。

> 训练时间和Loss走势（仅供参考）

Pretrain [512+8] & [768+16]
![input](./images/pretrain_loss.png)

SFT [512+8] & [768+16]
![input](./images/sft_loss.png)

## Ⅲ 模型权重

(原生PyTorch`*.pth`权重文件) 下载地址：
([ModelScope](https://www.modelscope.cn/models/gongjy/MiniMind2-V-PyTorch) | [HuggingFace](https://huggingface.co/jingyaogong/MiniMind2-V-PyTorch))

(`Transformers`格式模型)
下载地址：
([ModelScope](https://www.modelscope.cn/profile/gongjy) | [HuggingFace](https://huggingface.co/collections/jingyaogong/minimind-v-67000833fb60b3a2e1f3597d))

> 注：Transformers版本均为单图指令微调后的`MiniMind-V`模型

# 📌 Test

### 效果测试

#### 单图对话

<table>
  <thead>
    <tr>
      <th>图片</th>
      <th>MiniMind2-V</th>
      <th>MiniMind2-V-Small</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <img src="./dataset/eval_images/城市车水马龙-city-traffic.jpg" alt="city-traffic">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>图中是一个繁忙的城市街道，一条长长的街道两旁都是高楼大厦。这条街上挤满了汽车、卡车和公共汽车，还有许多其他车辆在路上行驶。在街道上，可以看到许多汽车，有的在高速行驶，而其他的则停在街道一侧。此外还有一辆公交车也停在街道的右侧。街道上可以看到交通灯，表明这是一个繁忙的城市环境。</td>
      <td>图中是一个繁忙的城市景象，有几辆汽车和一辆卡车行驶在城市街道上。可以看到许多交通信号灯，其中一些位于街道左侧，另一些则在右侧。可以看到有几个人在街上行走，其中一些人站得离街道更近一些，而另一些则距离较远。还有一个停车标志位于画面的左侧，暗示着城市环境。可以看到街道上有两辆汽车，一辆在右边，另一辆在左边，还有一辆在左边。这幅图像捕捉到了都市环境中典型的一天。</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/太空宇航员-Astronaut-Space.jpg" alt="astronaut">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>图片显示了一个宇航员的宇航员身穿宇航服，坐在一架大型航天飞机上。他们似乎正在进行一次宇航员登机或下机的旅程。在宇航员的身后，有一个火箭发射架，可能是用来支撑宇航员在旅程中的任务。此外，还有一架飞机停在机库附近，进一步表明这是一次航空展。在飞机的周围，还有一些人，但他们看起来离飞机很近。可以看到一个人站在飞机附近，可能正在观察或等待航天飞机准备起飞。</td>
      <td>场景中，一名士兵戴着头盔站在一架大型飞机上。这架飞机似乎是一架军用军用飞机，似乎正准备登上一架飞机。另一个人则站在前面，可能正在观察飞行过程。在飞机周围，有几个人，其中一些站在左侧，另一些则站在右侧。他们似乎正在观看飞行员的表现。此外，还有一辆卡车停在靠近左侧的位置，可能是为了更具体地观察飞行过程。</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/小狗美女海边-Dog-Woman-Sea.jpg" alt="dog-woman-sea">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>图片中，一个女人坐在沙滩上，手里拿着一只白色的狗。她看起来像是个女人，坐在沙地上，看着她。一只狗也坐在她旁边，看起来很放松和舒适。海滩上散布着其他沙滩游客，有些人坐着，而另一些人则坐在更远的地方。背景中可以看到一艘船，这表明这是一个受欢迎的海滩旅游目的地。</td>
      <td>两个人坐在海滩上，一边懒洋洋地躺在沙滩上，另一边则坐着。他们似乎正在享受海边时光。海滩上有几把椅子，其中一把靠近沙滩的左侧，另一把在中间。此外，还有一只狗躺在沙地上，为这个场景增添了一种放松的气氛。</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/彩虹瀑布-Rainbow-Falls.jpg" alt="rainbow-falls">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>照片捕捉到一幅美丽如画的大自然场景，背景是高山峦崖。在水边，一座巨大的喷泉横跨着水面，吸引着许多游客。水面上有几个人，他们或站或坐在喷泉周围，或站或坐。有些人可以看到他们在水中行走，而其他人则站在水边。总体而言，这幅画描绘的是一个美丽而宁静的环境，在那里人们可以欣赏到如画般的美景。</td>
      <td>在一个美丽的蓝色天空下，一座巨大而巨大的白色瀑布上方悬挂着一只巨大的湿流水。这只瀑布位于一座山上，为整个场景增添了一种迷人而又宁静的气氛。在这幅图像的背景中，可以看到几艘船，其中一些靠近水边，其他的则离得较远。这些船只似乎正在为风景或户外活动做准备。</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/椅子老人看书-Chair-Elderly-Reading.jpg" alt="elderly-reading">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>图中，一个男人坐在公园的长椅上，旁边是一把绿色椅子。他身边有一本打开的书，上面写着"读书"一句话，暗示他可能正在阅读。公园里有一张长椅和一张公园长椅，为周围的环境增添了几分生气。在公园的周围，有几辆汽车和一辆卡车，表明这是一个公共区域。此外，还可以看到一个人站在公园的不同位置上，可能是等着上路或过马路。</td>
      <td>一个穿着短裤的老人坐在公园长椅上，周围是树木。他似乎正在读一本书，可能是在读书。背景中有一座长凳，为这个场景提供了充足的座位。在背景中，可以看到一把椅子和一张餐桌，这说明这个场景可能是在一个户外座位区，那里有椅子供人们坐下来放松。</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/熊猫草地-Panda-Grassland.jpg" alt="panda-grassland">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>图中，一只白色的棕熊坐在草地上，旁边是一只长着棕色斑点的大熊。这只熊看起来很害羞或顽皮，因为它正躺在草地上休息，看上去很放松。</td>
      <td>在这幅图像中，一只棕色的熊正在草地上漫步。这只熊被放置在草地上，占据了画面的大部分空间。它似乎正在自然环境中行走，可能是在草地上。在背景中，有几棵树，为画面增添了自然元素。一只鸟在场景的中间附近飞翔，为画面增添了生气勃勃的气氛。</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/自行车鲜花-Bicycle-Flowers.jpg" alt="bicycle-flowers">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>图片展示了一个漂亮的花瓶，里面插满了五颜六色的鲜花和花束。这些花束散落在整个花瓶中，给人一种赏心悦目的感觉。花瓶里插着五颜六色鲜花，创造出一种令人赏心悦目的景象。这些鲜花被摆放在一张桌子上，很可能是为了展示它们的美丽而摆放的。</td>
      <td>场景中，一辆绿色和紫色相间的自行车停在一栋建筑旁边，它被放置在一棵大树旁。这辆自行车被摆放在附近，为这个场景增添了几分色彩。除了自行车外，还有一些其他的自行车，包括两个位于前景中的一个和位于背景中靠近中心位置的另一个。自行车的存在表明它可能是停在那里的。</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/舞蹈-dance.jpg" alt="dance">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>图片中的女人穿着一件白色连衣裙，脚上还系着一条黑色网球裙。她正在表演一个网球比赛，很可能是在比赛中。在背景中可以看到几把椅子，可能是为了观众或其他观众的座位安排而摆放的。此外，还有一个长凳放在场景左侧，为人们提供了一个休息的地方。</td>
      <td>一名身穿白色衣服的女子站在舞台上，手里拿着一只手拿着白色飞盘。她似乎正在参加一个舞台舞会或比赛。场景中还有其他几个人，其中一个站在舞台左侧，另一个站在右侧，第三个人则站在场地右侧。舞台上有几个观众，有的站着，有的坐着，还有一些站着。这看起来像是一场欢乐的节日或活动。</td>
    </tr>
  </tbody>
</table>

#### 多图对话（效果十分有限）

<table>
  <thead>
    <tr>
      <th>图片1</th>
      <th>图片2</th>
      <th>512_sft_multi</th>
      <th>768_sft_multi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><img src="./dataset/eval_multi_images/bird/0.jpg" alt="a-bird.png"></td>
      <td><img src="./dataset/eval_multi_images/bird/1.jpg" alt="a-bird.png"></td>
      <td>这幅图像显示了一种鸟簸戮的场景：一个女人站在红绿相间的红绿相间的紫色鸟簸戴在她身上。女人站在红色的鸟簸戴在她身上，而她的翻领上的那只红鸟则站在她身后。</td>
      <td>这两只鸟在同一片树林中飞翔，有的位于画面中心，而另一些则较小，形成了鲜明对比。这种鸟类的出现突出了其飞行能力和适应性，因为它们能够在树林中快速迅速移动。此外，两只鸟的位置不同，一个在图像的左边，另一个在右边，这表明它们在同一片树林中移动得很近。这种鸟类的自然行为也有助于区分这两种鸟类物种。</td>
    </tr>
  </tbody>
</table>

### 效果小结：

视觉信号对于LLM视作一种特殊的外语，
因此“学习外语”的能力高低，
很大程度上取决于LLM的能力。
LLM性能越强，对应的VLM必然越强，此时效果增益会很明显。

#### 未来值得改进的方面：

```text
> 更简单的Projection的跨模态特征对齐方式，相较于Cross-Attention可能处于劣势。
> Clip模型可以尝试更大性能更强的large系列，用更具细粒度的token表征图像特征，目前仍粗糙。
> 分辨率不高，理论上只有224×224（minimind-v数据集为节省空间，仅设定为128×128）。
> ...
```

# 📌 Acknowledge

> [!TIP]
> 如果您觉得 `MiniMind-V`对您有所帮助，可以在 GitHub 上加一个⭐<br/>
> 水平有限难免存在未知的纰漏，欢迎所有人在Issues交流指正或提交PR改进项目<br/>
> 您的支持就是持续改进项目的动力，谢谢！

## 🤝[贡献者](https://github.com/jingyaogong/minimind/graphs/contributors)

<a href="https://github.com/jingyaogong/minimind/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jingyaogong/minimind-v" />
</a>

## 😊鸣谢

<a href="https://github.com/xinyanghuang7"><b>@xinyanghuang7</b></a>:
<a href="https://github.com/xinyanghuang7/minimind-v/tree/hxy">🔗实现了完整的多图分支</a>

<details close> 
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

# 🎓 Citation

If you find MiniMind-V helpful in your research or work, please cite:

```bibtex
@misc{minimind,
  title={MiniMind-V: Train a Tiny VLM from scratch},
  author={Jingyao Gong},
  year={2024},
  howpublished={https://github.com/jingyaogong/minimind-v}
}
```

# License

This repository is licensed under the [Apache-2.0 License](LICENSE).
