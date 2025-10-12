# MiniMind-V Documentation Migration Summary

## Changes Made

### 1. Configuration Files
- **mkdocs.yml**: Updated site name, description, and URL from "MiniMind" to "MiniMind-V"

### 2. Images
- **Removed**: All old MiniMind LLM-related images
- **Added**: MiniMind-V VLM-specific images from `/images/`:
  - logo.png
  - minimind2-v.gif
  - VLM-structure.png
  - VLM-structure-moe.png
  - llava-structure.png
  - minimind-v-input.png
  - pretrain_loss.png
  - sft_loss.png

### 3. Documentation Files (All in English)

#### index.md
- Changed from LLM introduction to VLM introduction
- Updated model list to show MiniMind-V variants
- Added VLM-specific concepts and architecture
- Updated related links to minimind-v repository

#### quickstart.md
- Updated from LLM quick start to VLM quick start
- Added CLIP model download instructions
- Changed evaluation script from `eval_model.py` to `eval_vlm.py`
- Updated model architecture section with Visual Encoder details
- Added image-text dialogue examples

#### training.md
- Changed from LLM training to VLM training
- Added VLM dataset formats (pretrain_vlm_data, sft_vlm_data, sft_vlm_data_multi)
- Updated training scripts to `train_pretrain_vlm.py` and `train_sft_vlm.py`
- Added Visual Encoder and Projection layer explanations
- Updated training pipeline with VLM-specific steps
- Added multi-image fine-tuning section

#### README.md
- Updated project description to MiniMind-V
- Changed documentation URL to minimind-v.readthedocs.io
- Updated GitHub links to minimind-v repository

## Key Differences: MiniMind vs MiniMind-V

| Aspect | MiniMind (LLM) | MiniMind-V (VLM) |
|--------|----------------|------------------|
| Type | Language Model | Vision-Language Model |
| Input | Text only | Text + Images |
| Components | Transformer Decoder | Transformer Decoder + Visual Encoder + Projection |
| Training Data | Text (pretrain.jsonl, sft.jsonl) | Text + Images (pretrain_vlm_data, sft_vlm_data) |
| Eval Script | eval_model.py | eval_vlm.py |
| Training Scripts | train_pretrain.py, train_full_sft.py | train_pretrain_vlm.py, train_sft_vlm.py |
| Model Weights | *.pth (LLM only) | *.pth (LLM + Visual components) |

## Verification

All documentation now correctly references:
- ✅ MiniMind-V project name
- ✅ VLM-specific architecture diagrams
- ✅ Vision-language training pipeline
- ✅ Correct repository URLs (minimind-v)
- ✅ VLM evaluation and inference scripts
- ✅ Image-text dataset formats
- ✅ English language throughout

## Next Steps

1. Deploy to ReadTheDocs at minimind-v.readthedocs.io
2. Ensure .readthedocs.yaml is properly configured
3. Test local build with `mkdocs serve`
4. Verify all image links work correctly
