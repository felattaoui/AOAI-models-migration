# Azure OpenAI Models Migration Guide

Technical guide for migrating from GPT-4o/GPT-4o-mini to GPT-5.1/GPT-4.1-mini on Azure OpenAI.

## Migration Paths

| Current Model | Target Model | Retirement Date |
|---------------|--------------|-----------------|
| GPT-4o (2024-05-13) | GPT-5.1 | March 31, 2026 |
| GPT-4o (2024-08-06) | GPT-5.1 | March 31, 2026 |
| GPT-4o (2024-11-20) | GPT-5.1 | June 5, 2026 |
| GPT-4o-mini (2024-07-18) | GPT-4.1-mini | March 31, 2026 |

## Key API Changes

### Client Configuration

**Before (GPT-4o with versioned API):**
```python
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_ad_token_provider=token_provider,
    api_version="2024-02-15-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)
```

**After (GPT-5.1/GPT-4.1-mini with v1 API):**
```python
from openai import OpenAI

client = OpenAI(
    api_key=token_provider(),
    base_url=f"{AZURE_OPENAI_ENDPOINT}/openai/v1/"
)
```

### Parameter Changes

| Old Parameter | New Parameter | Notes |
|---------------|---------------|-------|
| `max_tokens` | `max_completion_tokens` | Required change for new models |
| `system` role | `developer` role | For reasoning models |

### GPT-5.1 Specific

- `reasoning_effort` parameter (defaults to `"none"`)
- `temperature`, `top_p`, `presence_penalty`, `frequency_penalty` **not supported**
- Use `developer` role instead of `system` role

### GPT-4.1-mini Specific

- Supports `temperature` and other sampling parameters
- More cost-effective than GPT-5.1 for simpler tasks

## Prerequisites

1. **Azure OpenAI Resource** with access to GPT-5.1 or GPT-4.1-mini models
2. **Azure CLI** installed and authenticated:
   ```bash
   az login
   ```
3. **Python packages**:
   ```bash
   pip install openai azure-identity
   ```

## Authentication

This guide uses **Microsoft Entra ID** authentication (recommended):

```python
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default"
)
```

## Usage

1. Open `azure_openai_migration_technical.ipynb` in Jupyter or VS Code
2. Set your `AZURE_OPENAI_ENDPOINT` in the configuration cell
3. Run `az login` to authenticate
4. Execute cells sequentially

## Contents

The notebook covers:
- Client configuration for both API versions
- Parameter mapping and changes
- Streaming responses
- Structured outputs (JSON mode)
- Function calling migration
- Prompt engineering best practices for new models

## Official Documentation

- [Azure OpenAI Model Retirements](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/concepts/model-retirements)
- [Azure OpenAI GPT-5.1 Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models)
- [OpenAI GPT-5 Prompting Guide](https://developers.openai.com/cookbook/examples/gpt-5/gpt-5_prompting_guide)

## License

MIT
