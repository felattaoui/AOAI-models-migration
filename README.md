# Azure OpenAI Models Migration Guide

Complete guide for migrating from GPT-4o/GPT-4o-mini to newer Azure OpenAI models, with **evaluation tools** to validate quality before deploying.

## Migration Paths

| Source Model | Target Model | Type | Use Case |
|--------------|--------------|------|----------|
| GPT-4o | **GPT-4.1** | Standard | Low-latency, high-throughput, drop-in replacement |
| GPT-4o | **GPT-5.1** | Reasoning | Official auto-migration target, built-in reasoning |
| GPT-4o | **GPT-5** | Reasoning | Latest model, configurable thinking levels |
| GPT-4o-mini | **GPT-4.1-mini** | Standard | Official auto-migration target |
| GPT-4o-mini | **GPT-5-mini** | Reasoning | Alternative with reasoning (higher cost) |

### How to Choose?

| Priority | GPT-4o replacement | GPT-4o-mini replacement |
|----------|-------------------|------------------------|
| **Low latency / high throughput** | GPT-4.1 | GPT-4.1-mini |
| **Balanced (cost + quality)** | GPT-5.1 | GPT-4.1-mini |
| **Best reasoning / agentic** | GPT-5 | GPT-5-mini |
| **Lowest cost** | GPT-4.1 | GPT-4.1-mini |

## Retirement Timeline (Updated February 2026)

| Deployment Type | GPT-4o (05-13, 08-06) | GPT-4o (11-20) | GPT-4o-mini |
|----------------|------------------------|-----------------|-------------|
| **Standard** | 2026-03-31 (auto-upgrade starts 03-09) | 2026-10-01 | 2026-03-31 |
| **Provisioned / Global / DataZone** | 2026-10-01 | 2026-10-01 | 2026-10-01 |

> Source: [Azure OpenAI Model Retirements](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/concepts/model-retirements)

## Repository Structure

```
.
├── azure_openai_migration_technical.ipynb   # Technical migration guide (API changes, code)
├── azure_openai_pricing_analysis.ipynb      # Cost calculators and pricing comparison
├── azure_openai_evaluation_guide.ipynb      # Evaluation demo notebook
├── src/                                      # Reusable Python modules
│   ├── __init__.py
│   ├── config.py                            # Model registry, migration paths, config
│   ├── clients.py                           # Client factory, unified call_model()
│   └── evaluate/                            # Evaluation framework
│       ├── __init__.py
│       ├── core.py                          # MigrationEvaluator, LLM-as-Judge, reports
│       ├── local_eval.py                    # Local SDK evaluation (azure-ai-evaluation)
│       ├── foundry.py                       # Azure AI Foundry cloud evaluation
│       ├── prompts/                         # System prompts in Prompty format (.prompty)
│       │   ├── __init__.py                  # load_prompty() / list_prompty() loader
│       │   ├── rag.prompty                  # RAG system prompt
│       │   ├── tool_calling.prompty         # Tool calling system prompt
│       │   ├── translate_*.prompty          # Translation prompts (fr_en, en_fr, en_de, technical)
│       │   └── classify_*.prompty           # Classification prompts (sentiment, category, intent, priority)
│       └── scenarios/                       # Pre-built evaluation scenarios
│           ├── __init__.py
│           ├── rag.py                       # RAG: groundedness, relevance, coherence
│           ├── tool_calling.py              # Tool calling: accuracy, parameters
│           ├── translation.py              # Translation: fluency, semantic equivalence
│           └── classification.py           # Classification: accuracy, consistency
├── requirements.txt
├── .env_example
└── README.md
```

## Key API Changes

### Client Configuration

**Before (GPT-4o — versioned API):**
```python
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_ad_token_provider=token_provider,
    api_version="2024-02-15-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)
```

**After (GPT-4.1 / GPT-5 series — v1 API):**
```python
from openai import OpenAI

client = OpenAI(
    api_key=token_provider(),
    base_url=f"{AZURE_OPENAI_ENDPOINT}/openai/v1/"
)
```

### Parameter Changes

| Parameter | GPT-4o | GPT-4.1 | GPT-5 / GPT-5.1 |
|-----------|--------|---------|-----------------|
| `max_tokens` | Supported | Use `max_completion_tokens` | Use `max_completion_tokens` |
| `temperature` | Supported | Supported | **Not supported** |
| `top_p` | Supported | Supported | **Not supported** |
| `reasoning_effort` | N/A | N/A | See table below |
| System role | `system` | `system` | `developer` |

### Reasoning Effort by Model

> **Important:** `reasoning_effort="none"` is only supported from GPT-5.1 onwards. GPT-5, GPT-5-mini, and GPT-5-nano do **not** support `"none"` — their minimum is `"minimal"`, which still incurs reasoning tokens and added latency. This is a key consideration when migrating from a non-reasoning model like GPT-4o.

| Model | Type | `reasoning_effort` levels | Default |
|-------|------|--------------------------|---------|
| GPT-4.1 / 4.1-mini / 4.1-nano | Standard | N/A (no reasoning) | — |
| GPT-5 / 5-mini / 5-nano | Reasoning | `minimal`, `low`, `medium`, `high` | `medium` |
| GPT-5.1 | Reasoning | `none`, `low`, `medium`, `high` | `none` |

## Evaluation

The evaluation framework helps you **detect regressions** before deploying a new model in production. It works by running the same prompts through both models and comparing the outputs.

### Pre-built Scenarios

Each scenario includes sample test data and adapted metrics — ready to run:

| Scenario | Metrics | Test Cases |
|----------|---------|------------|
| **RAG** | Groundedness, Relevance, Coherence | 8 examples (company docs, financial, legal) |
| **Tool Calling** | Tool Accuracy, Parameter Accuracy | 8 examples (weather, calendar, email, DB) |
| **Translation** | Fluency, Coherence, Relevance | 10 examples (FR/EN/DE, business/technical/legal) |
| **Classification** | Accuracy, Consistency, Relevance | 16 examples (sentiment, tickets, intent, priority) |

### Quick Start

```python
from src.evaluate.scenarios import create_rag_evaluator

# Compare GPT-4o vs GPT-4.1 on RAG tasks
evaluator = create_rag_evaluator(
    source_model="gpt-4o",
    target_model="gpt-4.1",
)
report = evaluator.run()
report.print_report()
```

### Cloud Evaluation (Azure AI Foundry)

```python
from src.evaluate.foundry import FoundryEvaluator

foundry = FoundryEvaluator(project_endpoint="https://your-project.services.ai.azure.com/...")
results = foundry.evaluate_cloud(report, evaluators=["coherence", "fluency", "relevance"])
```

See [azure_openai_evaluation_guide.ipynb](azure_openai_evaluation_guide.ipynb) for the full walkthrough.

## Prerequisites

1. **Azure OpenAI Resource** with access to target models
2. **Azure CLI** installed and authenticated:
   ```bash
   az login
   ```
3. **Python packages**:
   ```bash
   pip install -r requirements.txt
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

1. Copy `.env_example` to `.env` and fill in your values
2. Run `az login` to authenticate
3. Start with the **technical guide** notebook for API migration
4. Use the **evaluation guide** notebook to validate quality
5. Check the **pricing notebook** for cost analysis

## Official Documentation

- [Azure OpenAI Model Retirements](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/concepts/model-retirements)
- [Azure OpenAI Models Overview](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models)
- [GPT-5 vs GPT-4.1: Choosing the Right Model](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/gpt-5-vs-gpt-41)
- [Azure AI Foundry Evaluation](https://learn.microsoft.com/en-us/azure/ai-foundry/evaluation/)

## License

MIT
