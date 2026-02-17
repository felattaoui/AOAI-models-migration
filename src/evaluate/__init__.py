"""
Evaluation toolkit for Azure OpenAI model migration.

Two evaluation approaches:
- Local (quick prototyping): azure-ai-evaluation SDK via local_eval.py
- Cloud (full A/B testing): Foundry Evals API via foundry.py

Pre-built scenarios for common use cases: RAG, tool calling, translation, classification.
"""

from src.evaluate.core import (
    MigrationEvaluator,
    ComparisonReport,
    TestCase,
    EvalResult,
    load_test_cases,
)
from src.evaluate.foundry import (
    FoundryEvalsClient,
    WorkflowTrajectoryLog,
    FOUNDRY_METRICS,
    TEXT_QUALITY_METRICS,
    AGENT_QUALITY_METRICS,
    TOOL_USAGE_METRICS,
    ALL_FOUNDRY_METRICS,
    FOUNDRY_AVAILABLE,
)
from src.evaluate.local_eval import (
    quick_evaluate,
    evaluate_single,
    compare_local,
    get_model_config,
    SDK_AVAILABLE,
)

__all__ = [
    # Core evaluation
    "MigrationEvaluator",
    "ComparisonReport",
    "TestCase",
    "EvalResult",
    "load_test_cases",
    # Local SDK evaluation (quick prototyping)
    "quick_evaluate",
    "evaluate_single",
    "compare_local",
    "get_model_config",
    "SDK_AVAILABLE",
    # Foundry cloud evaluation (full A/B testing)
    "FoundryEvalsClient",
    "WorkflowTrajectoryLog",
    "FOUNDRY_METRICS",
    "TEXT_QUALITY_METRICS",
    "AGENT_QUALITY_METRICS",
    "TOOL_USAGE_METRICS",
    "ALL_FOUNDRY_METRICS",
    "FOUNDRY_AVAILABLE",
]
