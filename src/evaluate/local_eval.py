"""
Local evaluation using azure-ai-evaluation SDK.

Quick local evaluation for prototyping. Uses Microsoft's built-in
evaluators (same algorithms as Foundry cloud) for consistent scoring.

Two authentication modes:
- Entra ID (default): uses DefaultAzureCredential, no API key needed
- API key: pass api_key explicitly

Usage:
    from src.evaluate.local_eval import quick_evaluate, get_model_config

    model_config = get_model_config()
    result = quick_evaluate(
        items=[{"query": "What is Azure?", "response": "Azure is..."}],
        metrics=["coherence", "fluency"],
        model_config=model_config,
    )
    print(result["metrics"])
"""

import json
import logging
import os
import tempfile
from typing import Any, Optional

from dotenv import load_dotenv

# Suppress verbose SDK logs (promptflow execution.bulk, Run Summary, etc.)
os.environ["PF_LOGGING_LEVEL"] = "CRITICAL"
for _logger_name in (
    "azure.ai.evaluation",
    "promptflow",
    "execution",
    "execution.bulk",
    "azure.core.pipeline.policies.http_logging_policy",
):
    logging.getLogger(_logger_name).setLevel(logging.CRITICAL)


def _out(text):
    """Display output — uses IPython display in notebooks (VS Code fix), falls back to print."""
    try:
        from IPython.display import display, Markdown
        display(Markdown(str(text)))
    except ImportError:
        print(text)


def _find_metric(metrics_dict: dict, metric: str) -> float:
    """Find a metric value in SDK result, handling different key formats.

    The SDK evaluate() returns metrics with prefixed keys like
    "coherence.coherence" instead of just "coherence".
    """
    # Try exact key first
    if metric in metrics_dict:
        return metrics_dict[metric]
    # Try prefixed format: "coherence.coherence", "f1_score.f1_score"
    prefixed = f"{metric}.{metric}"
    if prefixed in metrics_dict:
        return metrics_dict[prefixed]
    # Try mean_ prefix
    if f"mean_{metric}" in metrics_dict:
        return metrics_dict[f"mean_{metric}"]
    # Fallback: search for any key containing the metric name
    for key, val in metrics_dict.items():
        if metric in key and isinstance(val, (int, float)):
            return val
    return 0


# ---------------------------------------------------------------------------
# Model config builder
# ---------------------------------------------------------------------------

def get_model_config(
    endpoint: str | None = None,
    deployment: str | None = None,
    api_key: str | None = None,
) -> dict:
    """
    Build model_config dict for SDK evaluators.

    Authentication:
      - Default: Entra ID via DefaultAzureCredential (generates a token
        and passes it as api_key — the SDK accepts this transparently)
      - If api_key is provided, uses it directly

    Args:
        endpoint: Azure OpenAI endpoint. Falls back to AZURE_OPENAI_ENDPOINT env var.
        deployment: Evaluator model deployment. Falls back to EVAL_MODEL_DEPLOYMENT env var.
        api_key: API key (optional). If not provided, Entra ID is used.

    Returns:
        Dict compatible with azure-ai-evaluation SDK evaluators.
    """
    load_dotenv()

    endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = deployment or os.getenv("EVAL_MODEL_DEPLOYMENT", "gpt-4.1")

    if not endpoint:
        raise ValueError("endpoint required (or set AZURE_OPENAI_ENDPOINT)")

    if api_key is None:
        # Entra ID: get a bearer token and pass it as api_key
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider

        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default",
        )
        api_key = token_provider()

    return {
        "azure_endpoint": endpoint,
        "azure_deployment": deployment,
        "api_key": api_key,
    }


# ---------------------------------------------------------------------------
# Evaluator registry
# ---------------------------------------------------------------------------

def _get_evaluator_class(metric: str):
    """Lazy-import evaluator class by metric name."""
    from azure.ai.evaluation import (
        CoherenceEvaluator,
        FluencyEvaluator,
        RelevanceEvaluator,
        GroundednessEvaluator,
        SimilarityEvaluator,
        F1ScoreEvaluator,
    )

    # LLM-as-judge evaluators (require model_config)
    llm_evaluators = {
        "coherence": CoherenceEvaluator,
        "fluency": FluencyEvaluator,
        "relevance": RelevanceEvaluator,
        "groundedness": GroundednessEvaluator,
        "similarity": SimilarityEvaluator,
    }

    # Deterministic evaluators (no model_config needed)
    deterministic_evaluators = {
        "f1_score": F1ScoreEvaluator,
    }

    registry = {**llm_evaluators, **deterministic_evaluators}
    cls = registry.get(metric)
    if not cls:
        raise ValueError(
            f"Unknown metric '{metric}'. Available: {list(registry.keys())}"
        )
    needs_model = metric in llm_evaluators
    return cls, needs_model


# ---------------------------------------------------------------------------
# Quick evaluate: single items
# ---------------------------------------------------------------------------

def evaluate_single(
    query: str,
    response: str,
    metrics: list[str],
    model_config: dict,
    context: str = "",
    ground_truth: str = "",
) -> dict[str, Any]:
    """
    Evaluate a single query/response pair locally.

    Returns dict of {metric_name: score, metric_name_reason: "..."}.

    Example:
        result = evaluate_single(
            query="What is Azure?",
            response="Azure is Microsoft's cloud platform.",
            metrics=["coherence", "fluency"],
            model_config=get_model_config(),
        )
        print(result)  # {"coherence": 5.0, "fluency": 4.0, ...}
    """
    results = {}
    for metric in metrics:
        cls, needs_model = _get_evaluator_class(metric)
        evaluator = cls(model_config=model_config) if needs_model else cls()

        kwargs = {"query": query, "response": response}
        if context and metric in ("groundedness", "relevance"):
            kwargs["context"] = context
        if ground_truth and metric in ("similarity", "f1_score"):
            kwargs["ground_truth"] = ground_truth

        result = evaluator(**kwargs)
        results.update(result)

    return results


# ---------------------------------------------------------------------------
# Quick evaluate: batch with SDK evaluate()
# ---------------------------------------------------------------------------

def quick_evaluate(
    items: list[dict],
    metrics: list[str],
    model_config: dict,
    azure_ai_project: str | None = None,
    output_path: str | None = None,
) -> dict:
    """
    Run SDK evaluate() locally on a list of items.

    Uses the same built-in evaluators as Foundry cloud. Results can
    optionally be logged to the Foundry portal via azure_ai_project.

    Args:
        items: List of dicts, each with at least "query" and "response".
               Optional: "context", "ground_truth".
        metrics: List of metric names (e.g., ["coherence", "fluency"]).
        model_config: From get_model_config().
        azure_ai_project: Foundry project URL for portal logging (optional).
        output_path: Path to save results JSON (optional).

    Returns:
        evaluate() result dict with "metrics" (averages) and "rows" (per-item).

    Example:
        items = [
            {"query": "What is Azure?", "response": "Azure is a cloud platform."},
            {"query": "What is Python?", "response": "Python is a language."},
        ]
        result = quick_evaluate(items, ["coherence", "fluency"], model_config)
        print(result["metrics"])  # {"coherence": 4.5, "fluency": 4.0}
    """
    from azure.ai.evaluation import evaluate

    # Build evaluators
    evaluators = {}
    for metric in metrics:
        cls, needs_model = _get_evaluator_class(metric)
        evaluators[metric] = cls(model_config=model_config) if needs_model else cls()

    # Write items to temp JSONL file (SDK requires file path)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        for item in items:
            clean = {k: v for k, v in item.items() if not k.startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")
        data_path = f.name

    # Build evaluate() kwargs
    eval_kwargs: dict[str, Any] = {
        "data": data_path,
        "evaluators": evaluators,
    }

    if azure_ai_project:
        eval_kwargs["azure_ai_project"] = azure_ai_project

    if output_path:
        eval_kwargs["output_path"] = output_path

    # Suppress verbose SDK output (promptflow spawns child processes that
    # print Run Summary / execution.bulk INFO lines to stdout/stderr).
    import io
    import contextlib

    _devnull = io.StringIO()
    os.environ["PF_LOGGING_LEVEL"] = "CRITICAL"

    with contextlib.redirect_stdout(_devnull), contextlib.redirect_stderr(_devnull):
        result = evaluate(**eval_kwargs)

    # Clean up temp file
    try:
        os.unlink(data_path)
    except OSError:
        pass

    return result


# ---------------------------------------------------------------------------
# Compare source vs target responses locally
# ---------------------------------------------------------------------------

def compare_local(
    source_items: list[dict],
    target_items: list[dict],
    metrics: list[str],
    model_config: dict,
    source_label: str = "source",
    target_label: str = "target",
) -> dict:
    """
    Compare source vs target model responses locally using SDK evaluate().

    Args:
        source_items: Items with source model responses.
        target_items: Items with target model responses.
        metrics: Metrics to evaluate.
        model_config: From get_model_config().
        source_label: Label for source model.
        target_label: Label for target model.

    Returns:
        {"source": result, "target": result, "summary": {metric: {source_avg, target_avg, delta}}}
    """
    _out(f"Evaluating **{source_label}** ({len(source_items)} items)...")
    source_result = quick_evaluate(source_items, metrics, model_config)

    _out(f"Evaluating **{target_label}** ({len(target_items)} items)...")
    target_result = quick_evaluate(target_items, metrics, model_config)

    # Build summary
    summary = {}
    source_metrics = source_result.get("metrics", {})
    target_metrics = target_result.get("metrics", {})

    for metric in metrics:
        src_avg = _find_metric(source_metrics, metric)
        tgt_avg = _find_metric(target_metrics, metric)
        summary[metric] = {
            "source_avg": round(float(src_avg), 2),
            "target_avg": round(float(tgt_avg), 2),
            "delta": round(float(tgt_avg) - float(src_avg), 2),
        }

    # Display summary as markdown table
    lines = [
        f"**Local SDK Comparison: {source_label} vs {target_label}**\n",
        f"| Metric | {source_label} | {target_label} | Delta |",
        "|--------|-------:|-------:|------:|",
    ]
    for metric, s in summary.items():
        lines.append(f"| {metric} | {s['source_avg']:.2f} | {s['target_avg']:.2f} | {s['delta']:+.2f} |")
    _out("\n".join(lines))

    return {
        "source": source_result,
        "target": target_result,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def check_sdk_available() -> bool:
    """Check if azure-ai-evaluation SDK is available."""
    try:
        from azure.ai.evaluation import evaluate  # noqa: F401
        return True
    except ImportError:
        return False


SDK_AVAILABLE = check_sdk_available()
