"""
LoRA (Low-Rank Adaptation) utilities for MiniMindVLM
Enables parameter-efficient fine-tuning with <2% trainable parameters
"""
import torch
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from typing import Optional


def get_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None,
    bias: str = "none",
) -> LoraConfig:
    """
    Get LoRA configuration for MiniMindVLM

    Args:
        r: Rank of LoRA update matrices (higher = more parameters)
        lora_alpha: Scaling factor for LoRA updates
        lora_dropout: Dropout rate for LoRA layers
        target_modules: Which modules to apply LoRA to. If None, uses default for MiniMind
        bias: How to handle bias parameters ("none", "all", "lora_only")

    Returns:
        LoraConfig object ready to apply to model

    Typical memory savings:
        - r=8:  ~0.5% trainable params, 99.5% frozen
        - r=16: ~1.0% trainable params, 99.0% frozen  (recommended)
        - r=32: ~2.0% trainable params, 98.0% frozen
    """

    # Default target modules for MiniMind architecture
    # These are the attention projection layers
    if target_modules is None:
        target_modules = [
            "q_proj",    # Query projection in attention
            "k_proj",    # Key projection in attention
            "v_proj",    # Value projection in attention
            "o_proj",    # Output projection in attention
        ]

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
    )

    return config


def apply_lora(model, lora_config: LoraConfig, verbose: bool = True):
    """
    Apply LoRA adapters to a MiniMindVLM model

    Args:
        model: MiniMindVLM model instance
        lora_config: LoRA configuration from get_lora_config()
        verbose: Print trainable parameter statistics

    Returns:
        Model with LoRA adapters applied
    """
    # Freeze vision encoder (it's already frozen in MiniMindVLM, but be explicit)
    if hasattr(model, 'vision_encoder') and model.vision_encoder is not None:
        for param in model.vision_encoder.parameters():
            param.requires_grad = False

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    if verbose:
        model.print_trainable_parameters()

        # Calculate memory savings
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        print(f"\n📊 LoRA Statistics:")
        print(f"  Total parameters: {total / 1e6:.2f}M")
        print(f"  Trainable parameters: {trainable / 1e6:.2f}M ({trainable / total * 100:.2f}%)")
        print(f"  Frozen parameters: {(total - trainable) / 1e6:.2f}M ({(total - trainable) / total * 100:.2f}%)")
        print(f"  Memory savings: ~{100 - (trainable / total * 100):.1f}%")

    return model


def load_lora_model(base_model, lora_adapter_path: str):
    """
    Load a model with LoRA adapters

    Args:
        base_model: Base MiniMindVLM model (without LoRA)
        lora_adapter_path: Path to saved LoRA adapters

    Returns:
        Model with LoRA adapters loaded
    """
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    return model


def merge_lora_weights(model, save_path: Optional[str] = None):
    """
    Merge LoRA weights into base model for deployment
    After merging, the model is a standard model without adapters

    Args:
        model: Model with LoRA adapters
        save_path: If provided, save merged model to this path

    Returns:
        Merged model (standard model, no adapters)
    """
    # Merge LoRA weights into base model
    model = model.merge_and_unload()

    if save_path:
        model.save_pretrained(save_path)
        print(f"✅ Merged model saved to {save_path}")

    return model


# Example usage:
if __name__ == "__main__":
    from model.model_vlm import MiniMindVLM, VLMConfig

    # Create model
    config = VLMConfig()
    model = MiniMindVLM(config)

    # Apply LoRA
    lora_config = get_lora_config(r=16)
    model = apply_lora(model, lora_config)

    # Now model is ready for training with LoRA
    # Only ~1% of parameters will be updated during training
