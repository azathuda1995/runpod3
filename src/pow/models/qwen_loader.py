"""Qwen model loader for PoC v2.

Loads Qwen model from HuggingFace with pretrained weights.
RunPod caches models on hosts via MODEL_NAME env variable.
"""
import os
import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoConfig

from common.logger import create_logger

logger = create_logger(__name__)

# Model name from environment or default
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8")

# Default k_dim for artifacts (first k dimensions of normalized logits)
K_DIM = int(os.getenv("K_DIM", "12"))


def load_qwen_model(
    model_name: str = MODEL_NAME,
    dtype: torch.dtype = torch.float16,
    device_map: str = "auto",
) -> torch.nn.Module:
    """
    Load Qwen model from HuggingFace with pretrained weights.

    Args:
        model_name: HuggingFace model ID (e.g., "Qwen/Qwen3-32B-FP8")
        dtype: Model dtype for computations
        device_map: Device mapping strategy ("auto" for multi-GPU)

    Returns:
        Loaded model ready for inference
    """
    logger.info(f"Loading Qwen model: {model_name}")
    start_time = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,  # Qwen requires this
    )

    model.eval()
    model.requires_grad_(False)

    load_time = time.time() - start_time
    logger.info(
        f"Qwen model loaded in {load_time:.2f}s: "
        f"hidden_size={model.config.hidden_size}, "
        f"num_layers={model.config.num_hidden_layers}, "
        f"vocab_size={model.config.vocab_size}"
    )

    return model


def load_qwen_model_to_device(
    model_name: str = MODEL_NAME,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float16,
) -> torch.nn.Module:
    """
    Load Qwen model to a specific device (single GPU).

    Args:
        model_name: HuggingFace model ID
        device: Target device (e.g., "cuda:0")
        dtype: Model dtype

    Returns:
        Model loaded on specified device
    """
    logger.info(f"Loading Qwen model to {device}: {model_name}")
    start_time = time.time()

    # Load to CPU first, then move to device
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map={"": device},  # All on one device
        trust_remote_code=True,
    )

    model.eval()
    model.requires_grad_(False)

    load_time = time.time() - start_time
    logger.info(f"Qwen model loaded to {device} in {load_time:.2f}s")

    return model


def get_qwen_config(model_name: str = MODEL_NAME) -> dict:
    """
    Get model configuration for parameter validation.

    Args:
        model_name: HuggingFace model ID

    Returns:
        Dict with model parameters
    """
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    return {
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "vocab_size": config.vocab_size,
        "intermediate_size": getattr(config, "intermediate_size", None),
    }


def get_qwen_vocab_size(model_name: str = MODEL_NAME) -> int:
    """Get vocab size for the model."""
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    return config.vocab_size
