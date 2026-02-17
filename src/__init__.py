"""
Azure OpenAI Migration Toolkit

Provides utilities for migrating from GPT-4o/GPT-4o-mini to newer models
(GPT-4.1, GPT-5.1, GPT-5 series) and evaluating migration quality.
"""

from src.config import MODEL_REGISTRY, load_config, ModelConfig
from src.clients import create_client, call_model

__all__ = [
    "MODEL_REGISTRY",
    "load_config",
    "ModelConfig",
    "create_client",
    "call_model",
]
