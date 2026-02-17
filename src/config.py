"""
Model configuration and registry for Azure OpenAI models.

Centralizes model metadata: type (reasoning vs standard), supported parameters,
retirement dates, and recommended migration targets.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv


@dataclass
class ModelConfig:
    """Configuration for an Azure OpenAI model."""

    name: str
    model_type: str  # "reasoning" or "standard"
    supports_temperature: bool
    supports_top_p: bool
    supports_reasoning_effort: bool
    max_tokens_param: str  # "max_tokens" or "max_completion_tokens"
    supports_system_role: bool = True
    supports_developer_role: bool = False
    default_reasoning_effort: Optional[str] = None
    min_reasoning_effort: Optional[str] = None  # Lowest allowed reasoning_effort value
    retirement_date_standard: Optional[str] = None
    retirement_date_provisioned: Optional[str] = None
    replacement_model: Optional[str] = None
    api_style: str = "v1"  # "classic" (AzureOpenAI) or "v1" (OpenAI with v1 endpoint)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, ModelConfig] = {
    # ── Source models (being retired) ──────────────────────────────────────
    "gpt-4o": ModelConfig(
        name="gpt-4o",
        model_type="standard",
        supports_temperature=True,
        supports_top_p=True,
        supports_reasoning_effort=False,
        max_tokens_param="max_tokens",
        api_style="classic",
        retirement_date_standard="2026-03-31",
        retirement_date_provisioned="2026-10-01",
        replacement_model="gpt-5.1",
        notes=[
            "Auto-upgrades for Standard deployments start 2026-03-09",
            "Provisioned/Global Standard/Data Zone Standard: retirement moved to 2026-10-01",
        ],
    ),
    "gpt-4o-mini": ModelConfig(
        name="gpt-4o-mini",
        model_type="standard",
        supports_temperature=True,
        supports_top_p=True,
        supports_reasoning_effort=False,
        max_tokens_param="max_tokens",
        api_style="classic",
        retirement_date_standard="2026-03-31",
        retirement_date_provisioned="2026-10-01",
        replacement_model="gpt-4.1-mini",
        notes=[
            "Auto-upgrades for Standard deployments start 2026-03-09",
            "Provisioned/Global Standard/Data Zone Standard: retirement moved to 2026-10-01",
        ],
    ),
    # ── GPT-4.1 series (standard, non-reasoning) ──────────────────────────
    "gpt-4.1": ModelConfig(
        name="gpt-4.1",
        model_type="standard",
        supports_temperature=True,
        supports_top_p=True,
        supports_reasoning_effort=False,
        max_tokens_param="max_completion_tokens",
        api_style="v1",
        retirement_date_standard="2026-10-14",
        replacement_model="gpt-5",
        notes=[
            "Optimized for speed and throughput",
            "Up to 1M token context (long-context variant)",
            "Known issue: tool definitions >300K tokens may fail",
        ],
    ),
    "gpt-4.1-mini": ModelConfig(
        name="gpt-4.1-mini",
        model_type="standard",
        supports_temperature=True,
        supports_top_p=True,
        supports_reasoning_effort=False,
        max_tokens_param="max_completion_tokens",
        api_style="v1",
        retirement_date_standard="2026-10-14",
        replacement_model="gpt-5-mini",
    ),
    "gpt-4.1-nano": ModelConfig(
        name="gpt-4.1-nano",
        model_type="standard",
        supports_temperature=True,
        supports_top_p=True,
        supports_reasoning_effort=False,
        max_tokens_param="max_completion_tokens",
        api_style="v1",
        retirement_date_standard="2026-10-14",
        replacement_model="gpt-5-nano",
    ),
    # ── GPT-5.1 (reasoning) ───────────────────────────────────────────────
    "gpt-5.1": ModelConfig(
        name="gpt-5.1",
        model_type="reasoning",
        supports_temperature=False,
        supports_top_p=False,
        supports_reasoning_effort=True,
        supports_developer_role=True,
        max_tokens_param="max_completion_tokens",
        default_reasoning_effort="none",
        min_reasoning_effort="none",
        api_style="v1",
        notes=[
            "reasoning_effort defaults to 'none' - set explicitly if you want reasoning",
            "Supports 'none' as reasoning_effort for fast non-reasoning responses",
            "Supported levels: none, low, medium, high",
            "Does NOT support temperature, top_p, presence_penalty, frequency_penalty",
        ],
    ),
    # ── GPT-5 (reasoning) ─────────────────────────────────────────────────
    "gpt-5": ModelConfig(
        name="gpt-5",
        model_type="reasoning",
        supports_temperature=False,
        supports_top_p=False,
        supports_reasoning_effort=True,
        supports_developer_role=True,
        max_tokens_param="max_completion_tokens",
        default_reasoning_effort="medium",
        min_reasoning_effort="minimal",
        api_style="v1",
        notes=[
            "Supported levels: minimal, low, medium, high (NOT 'none')",
            "Parallel tool calls NOT supported at minimal reasoning_effort",
            "272K tokens in, 128K tokens out (400K total)",
        ],
    ),
    "gpt-5-mini": ModelConfig(
        name="gpt-5-mini",
        model_type="reasoning",
        supports_temperature=False,
        supports_top_p=False,
        supports_reasoning_effort=True,
        supports_developer_role=True,
        max_tokens_param="max_completion_tokens",
        default_reasoning_effort="medium",
        min_reasoning_effort="minimal",
        api_style="v1",
        notes=[
            "Supported levels: minimal, low, medium, high (NOT 'none')",
        ],
    ),
    "gpt-5-nano": ModelConfig(
        name="gpt-5-nano",
        model_type="reasoning",
        supports_temperature=False,
        supports_top_p=False,
        supports_reasoning_effort=True,
        supports_developer_role=True,
        max_tokens_param="max_completion_tokens",
        default_reasoning_effort="medium",
        min_reasoning_effort="minimal",
        api_style="v1",
        notes=[
            "Supported levels: minimal, low, medium, high (NOT 'none')",
            "Smallest and cheapest GPT-5 family model",
        ],
    ),
}

# ---------------------------------------------------------------------------
# Migration paths
# ---------------------------------------------------------------------------

MIGRATION_PATHS: dict[str, list[dict]] = {
    "gpt-4o": [
        {
            "target": "gpt-4.1",
            "type": "standard",
            "description": "Fast, low-latency, no reasoning. Best for real-time chat, Q&A, summarization.",
        },
        {
            "target": "gpt-5.1",
            "type": "reasoning",
            "description": "Official replacement. Deep reasoning, agentic workflows, code generation.",
        },
        {
            "target": "gpt-5",
            "type": "reasoning",
            "description": "Latest reasoning model. 4 thinking levels, best accuracy.",
        },
    ],
    "gpt-4o-mini": [
        {
            "target": "gpt-4.1-mini",
            "type": "standard",
            "description": "Official replacement. Cost-effective, fast responses.",
        },
        {
            "target": "gpt-5-mini",
            "type": "reasoning",
            "description": "Reasoning capabilities at mini model cost.",
        },
    ],
}


def load_config(env_path: Optional[str] = None) -> dict:
    """Load configuration from .env file and return deployment mappings."""
    load_dotenv(env_path or ".env")

    config = {
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "deployments": {},
    }

    deployment_vars = {
        "gpt-4o": "GPT4O_DEPLOYMENT",
        "gpt-4o-mini": "GPT4O_MINI_DEPLOYMENT",
        "gpt-4.1": "GPT41_DEPLOYMENT",
        "gpt-4.1-mini": "GPT41_MINI_DEPLOYMENT",
        "gpt-5.1": "GPT51_DEPLOYMENT",
        "gpt-5": "GPT5_DEPLOYMENT",
        "gpt-5-mini": "GPT5_MINI_DEPLOYMENT",
    }

    for model_name, env_var in deployment_vars.items():
        value = os.getenv(env_var)
        if value:
            config["deployments"][model_name] = value

    # Foundry evaluation config
    config["foundry_endpoint"] = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
    config["eval_model"] = os.getenv("EVAL_MODEL_DEPLOYMENT", "gpt-4.1")
    config["model_deployment_name"] = os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME", config["eval_model"])

    return config
