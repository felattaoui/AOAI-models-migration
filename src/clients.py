"""
Client factory and unified model calling for Azure OpenAI.

Handles the differences between classic (AzureOpenAI) and v1 (OpenAI) clients,
and translates parameters automatically based on model capabilities.

Authentication: Entra ID (DefaultAzureCredential) by default, API key as fallback.
"""

import os
from typing import Any, Optional

from openai import AzureOpenAI, OpenAI

from src.config import MODEL_REGISTRY, ModelConfig, load_config


def _get_token_provider():
    """Create an Entra ID token provider for Azure Cognitive Services."""
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    return get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )


def create_client(
    model_name: str,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    api_version: str = "2024-12-01-preview",
) -> AzureOpenAI | OpenAI:
    """
    Create the appropriate client for a given model.

    Authentication:
      - Default: Entra ID (DefaultAzureCredential)
      - Only uses API key if explicitly passed as api_key argument

    Client type:
      - Legacy models (gpt-4o, gpt-4o-mini): AzureOpenAI with api_version
      - New models (gpt-4.1+, gpt-5+): OpenAI with /openai/v1/ endpoint

    Args:
        model_name: Model identifier (e.g., "gpt-4o", "gpt-4.1", "gpt-5.1")
        endpoint: Azure OpenAI endpoint URL. Falls back to AZURE_OPENAI_ENDPOINT env var.
        api_key: API key (optional). If not provided, Entra ID is used.
        api_version: API version for classic clients.

    Returns:
        Configured OpenAI or AzureOpenAI client.
    """
    model_config = MODEL_REGISTRY.get(model_name)
    if not model_config:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )

    endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        raise ValueError("endpoint is required (or set AZURE_OPENAI_ENDPOINT)")

    # Entra ID by default. Only use API key if explicitly passed as argument.
    use_entra = api_key is None

    if model_config.api_style == "classic":
        if use_entra:
            return AzureOpenAI(
                azure_endpoint=endpoint,
                azure_ad_token_provider=_get_token_provider(),
                api_version=api_version,
            )
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
    else:
        base_url = endpoint.rstrip("/") + "/openai/v1"
        if use_entra:
            token_provider = _get_token_provider()
            return OpenAI(
                base_url=base_url,
                api_key=token_provider(),
            )
        return OpenAI(
            base_url=base_url,
            api_key=api_key,
        )


def _translate_params(model_config: ModelConfig, params: dict[str, Any]) -> dict[str, Any]:
    """Translate API parameters based on model capabilities."""
    translated = {}

    for key, value in params.items():
        # max_tokens → max_completion_tokens for new models
        if key == "max_tokens" and model_config.max_tokens_param == "max_completion_tokens":
            translated["max_completion_tokens"] = value
        elif key == "max_completion_tokens":
            translated["max_completion_tokens"] = value
        # Drop unsupported params for reasoning models
        elif key == "temperature" and not model_config.supports_temperature:
            continue
        elif key == "top_p" and not model_config.supports_top_p:
            continue
        elif key in ("presence_penalty", "frequency_penalty", "logprobs", "top_logprobs", "logit_bias"):
            if model_config.model_type == "reasoning":
                continue
            translated[key] = value
        else:
            translated[key] = value

    # Add reasoning_effort for reasoning models if not already set
    if model_config.supports_reasoning_effort and "reasoning_effort" not in translated:
        translated["reasoning_effort"] = model_config.default_reasoning_effort

    return translated


def call_model(
    client: AzureOpenAI | OpenAI,
    model_name: str,
    messages: list[dict[str, str]],
    deployment: Optional[str] = None,
    **params: Any,
) -> Any:
    """
    Call a model with automatic parameter translation.

    Handles differences between models:
    - Removes temperature/top_p for reasoning models
    - Converts max_tokens → max_completion_tokens
    - Adds reasoning_effort for reasoning models

    Args:
        client: OpenAI or AzureOpenAI client.
        model_name: Model identifier for parameter translation.
        messages: Chat messages.
        deployment: Deployment name (defaults to model_name).
        **params: Additional parameters (temperature, max_tokens, etc.)

    Returns:
        ChatCompletion response.
    """
    model_config = MODEL_REGISTRY.get(model_name)
    if not model_config:
        raise ValueError(f"Unknown model '{model_name}'")

    translated = _translate_params(model_config, params)
    model_id = deployment or model_name

    return client.chat.completions.create(
        model=model_id,
        messages=messages,
        **translated,
    )
