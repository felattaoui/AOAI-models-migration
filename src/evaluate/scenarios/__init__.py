"""
Pre-built evaluation scenarios for common use cases.

Each scenario provides:
- Sample test data (ready to use)
- Adapted metrics
- Complete evaluation code comparing old vs new model
"""

from src.evaluate.scenarios.rag import RAG_TEST_CASES, RAG_METRICS, create_rag_evaluator
from src.evaluate.scenarios.tool_calling import TOOL_CALLING_TEST_CASES, TOOL_CALLING_METRICS, create_tool_calling_evaluator
from src.evaluate.scenarios.translation import TRANSLATION_TEST_CASES, TRANSLATION_METRICS, create_translation_evaluator
from src.evaluate.scenarios.classification import CLASSIFICATION_TEST_CASES, CLASSIFICATION_METRICS, create_classification_evaluator

__all__ = [
    "RAG_TEST_CASES", "RAG_METRICS", "create_rag_evaluator",
    "TOOL_CALLING_TEST_CASES", "TOOL_CALLING_METRICS", "create_tool_calling_evaluator",
    "TRANSLATION_TEST_CASES", "TRANSLATION_METRICS", "create_translation_evaluator",
    "CLASSIFICATION_TEST_CASES", "CLASSIFICATION_METRICS", "create_classification_evaluator",
]
