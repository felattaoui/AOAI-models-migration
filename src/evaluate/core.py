"""
Core evaluation pipeline for A/B model comparison during migration.

Runs the same prompts through source and target models, evaluates outputs
using LLM-as-Judge, and generates a comparison report.
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import AzureOpenAI, OpenAI

from src.config import MODEL_REGISTRY, load_config
from src.clients import create_client, call_model


def _out(text):
    """Display output — uses IPython display in notebooks (VS Code fix), falls back to print."""
    try:
        from IPython.display import display, Markdown
        display(Markdown(str(text)))
    except ImportError:
        print(text)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    """A single test case for evaluation."""
    prompt: str
    system_prompt: str = "You are a helpful assistant."
    expected_output: Optional[str] = None
    context: Optional[str] = None  # For RAG scenarios
    tools: Optional[list[dict]] = None  # For tool-calling scenarios
    ground_truth_label: Optional[str] = None  # For classification scenarios
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result for a single test case comparing source vs target."""
    test_case: TestCase
    source_response: str
    target_response: str
    source_scores: dict[str, float]
    target_scores: dict[str, float]
    regression_detected: bool
    details: dict = field(default_factory=dict)


@dataclass
class ComparisonReport:
    """Aggregated comparison report across all test cases."""
    source_model: str
    target_model: str
    results: list[EvalResult]
    metrics: list[str]
    regression_threshold: float = -0.5

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def regressions(self) -> int:
        return sum(1 for r in self.results if r.regression_detected)

    @property
    def improvements(self) -> int:
        count = 0
        for r in self.results:
            avg_source = sum(r.source_scores.values()) / max(len(r.source_scores), 1)
            avg_target = sum(r.target_scores.values()) / max(len(r.target_scores), 1)
            if avg_target > avg_source + abs(self.regression_threshold):
                count += 1
        return count

    @property
    def equivalent(self) -> int:
        return self.total - self.regressions - self.improvements

    def avg_scores(self) -> dict[str, dict[str, float]]:
        """Average scores per metric for source and target."""
        source_totals: dict[str, float] = {}
        target_totals: dict[str, float] = {}
        counts: dict[str, int] = {}

        for r in self.results:
            for metric, score in r.source_scores.items():
                source_totals[metric] = source_totals.get(metric, 0) + score
                counts[metric] = counts.get(metric, 0) + 1
            for metric, score in r.target_scores.items():
                target_totals[metric] = target_totals.get(metric, 0) + score

        avg = {}
        for metric in counts:
            n = counts[metric]
            avg[metric] = {
                "source": round(source_totals.get(metric, 0) / n, 2),
                "target": round(target_totals.get(metric, 0) / n, 2),
                "delta": round(
                    (target_totals.get(metric, 0) - source_totals.get(metric, 0)) / n, 2
                ),
            }
        return avg

    def print_report(self) -> None:
        """Print a formatted comparison report to console."""
        print("=" * 70)
        print(f"  MIGRATION EVALUATION REPORT")
        print(f"  {self.source_model} → {self.target_model}")
        print("=" * 70)
        print()

        # Summary
        print(f"  Total test cases:  {self.total}")
        print(f"  Regressions:       {self.regressions}")
        print(f"  Improvements:      {self.improvements}")
        print(f"  Equivalent:        {self.equivalent}")
        print()

        # Average scores
        avg = self.avg_scores()
        print(f"  {'Metric':<25} {'Source':>8} {'Target':>8} {'Delta':>8}")
        print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
        for metric, scores in avg.items():
            delta_str = f"{scores['delta']:+.2f}"
            indicator = ""
            if scores["delta"] < self.regression_threshold:
                indicator = " ⚠ REGRESSION"
            elif scores["delta"] > abs(self.regression_threshold):
                indicator = " ✓ IMPROVED"
            print(
                f"  {metric:<25} {scores['source']:>8.2f} {scores['target']:>8.2f} {delta_str:>8}{indicator}"
            )
        print()

        # Individual results with regressions
        if self.regressions > 0:
            print("  REGRESSIONS DETECTED:")
            print(f"  {'-'*66}")
            for i, r in enumerate(self.results):
                if r.regression_detected:
                    prompt_preview = r.test_case.prompt[:60].replace("\n", " ")
                    print(f"  [{i}] {prompt_preview}...")
                    for metric in self.metrics:
                        src = r.source_scores.get(metric, 0)
                        tgt = r.target_scores.get(metric, 0)
                        if tgt < src + self.regression_threshold:
                            print(f"      {metric}: {src:.1f} → {tgt:.1f} ({tgt-src:+.1f})")
            print()

        # Verdict
        if self.regressions == 0:
            print("  ✓ MIGRATION SAFE: No regressions detected.")
        elif self.regressions <= self.total * 0.1:
            print("  ⚠ MIGRATION CAUTION: Minor regressions detected. Review flagged cases.")
        else:
            print("  ✗ MIGRATION RISKY: Significant regressions detected. Investigate before proceeding.")
        print("=" * 70)

    def to_dict(self) -> dict:
        """Export report as dictionary (JSON-serializable)."""
        return {
            "source_model": self.source_model,
            "target_model": self.target_model,
            "summary": {
                "total": self.total,
                "regressions": self.regressions,
                "improvements": self.improvements,
                "equivalent": self.equivalent,
            },
            "avg_scores": self.avg_scores(),
            "results": [
                {
                    "prompt": r.test_case.prompt[:100],
                    "source_response": r.source_response[:200],
                    "target_response": r.target_response[:200],
                    "source_scores": r.source_scores,
                    "target_scores": r.target_scores,
                    "regression": r.regression_detected,
                }
                for r in self.results
            ],
        }

    def save(self, path: str) -> None:
        """Save report to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"  Report saved to {path}")


# ---------------------------------------------------------------------------
# LLM-as-Judge evaluator
# ---------------------------------------------------------------------------

JUDGE_PROMPTS = {
    "coherence": (
        "Rate the coherence of the following response on a scale of 1 to 5.\n"
        "1 = Incoherent, contradictory, or nonsensical\n"
        "3 = Mostly coherent with minor issues\n"
        "5 = Perfectly coherent, logical, and well-structured\n\n"
        "Response to evaluate:\n{response}\n\n"
        "Return ONLY a JSON object: {{\"score\": <1-5>, \"reason\": \"<brief explanation>\"}}"
    ),
    "fluency": (
        "Rate the fluency of the following response on a scale of 1 to 5.\n"
        "1 = Poor grammar, awkward phrasing\n"
        "3 = Acceptable with minor issues\n"
        "5 = Natural, well-written, publication-quality\n\n"
        "Response to evaluate:\n{response}\n\n"
        "Return ONLY a JSON object: {{\"score\": <1-5>, \"reason\": \"<brief explanation>\"}}"
    ),
    "relevance": (
        "Rate how relevant the response is to the given query on a scale of 1 to 5.\n"
        "1 = Completely off-topic\n"
        "3 = Partially relevant\n"
        "5 = Directly addresses the query with appropriate detail\n\n"
        "Query: {query}\n\n"
        "Response to evaluate:\n{response}\n\n"
        "Return ONLY a JSON object: {{\"score\": <1-5>, \"reason\": \"<brief explanation>\"}}"
    ),
    "groundedness": (
        "Rate how well the response is grounded in the provided context on a scale of 1 to 5.\n"
        "1 = Contains fabricated information not in the context\n"
        "3 = Mostly grounded with some unsupported claims\n"
        "5 = Fully grounded, every claim is supported by the context\n\n"
        "Context:\n{context}\n\n"
        "Query: {query}\n\n"
        "Response to evaluate:\n{response}\n\n"
        "Return ONLY a JSON object: {{\"score\": <1-5>, \"reason\": \"<brief explanation>\"}}"
    ),
}


def _judge_response(
    judge_client: OpenAI | AzureOpenAI,
    judge_model: str,
    metric: str,
    response: str,
    query: str = "",
    context: str = "",
) -> dict[str, Any]:
    """Use LLM-as-Judge to score a response on a given metric."""
    prompt_template = JUDGE_PROMPTS.get(metric)
    if not prompt_template:
        return {"score": 0, "reason": f"Unknown metric: {metric}"}

    prompt = prompt_template.format(
        response=response,
        query=query,
        context=context,
    )

    try:
        result = judge_client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": "You are an impartial evaluation judge. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_completion_tokens=150,
        )
        content = result.choices[0].message.content.strip()
        # Parse JSON from response
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content)
    except (json.JSONDecodeError, Exception) as e:
        return {"score": 0, "reason": f"Judge error: {str(e)[:100]}"}


# ---------------------------------------------------------------------------
# Migration Evaluator
# ---------------------------------------------------------------------------

class MigrationEvaluator:
    """
    Evaluates migration quality by comparing source and target model outputs.

    Runs each test case through both models, scores outputs with LLM-as-Judge,
    and generates a comparison report.

    Usage:
        evaluator = MigrationEvaluator(
            source_model="gpt-4o",
            target_model="gpt-4.1",
            test_cases=[TestCase(prompt="Hello"), ...],
            metrics=["coherence", "fluency", "relevance"],
        )
        report = evaluator.run()
        report.print_report()
    """

    def __init__(
        self,
        source_model: str,
        target_model: str,
        test_cases: list[TestCase],
        metrics: list[str] | None = None,
        judge_model: str | None = None,
        regression_threshold: float = -0.5,
        endpoint: str | None = None,
        api_key: str | None = None,
        source_deployment: str | None = None,
        target_deployment: str | None = None,
        judge_deployment: str | None = None,
    ):
        self.source_model = source_model
        self.target_model = target_model
        self.test_cases = test_cases
        self.metrics = metrics or ["coherence", "fluency", "relevance"]
        self.judge_model = judge_model or os.getenv("EVAL_MODEL_DEPLOYMENT", "gpt-4.1")
        self.regression_threshold = regression_threshold
        self.source_deployment = source_deployment
        self.target_deployment = target_deployment
        self.judge_deployment = judge_deployment

        self.source_client = create_client(source_model, endpoint, api_key)
        self.target_client = create_client(target_model, endpoint, api_key)
        self.judge_client = create_client(self.judge_model, endpoint, api_key)

    def _call_source(self, test_case: TestCase) -> str:
        """Call source model with a test case."""
        messages = [
            {"role": "system", "content": test_case.system_prompt},
            {"role": "user", "content": test_case.prompt},
        ]
        response = call_model(
            self.source_client,
            self.source_model,
            messages,
            deployment=self.source_deployment,
            max_tokens=1024,
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

    def _call_target(self, test_case: TestCase) -> str:
        """Call target model with a test case."""
        messages = [
            {"role": "system", "content": test_case.system_prompt},
            {"role": "user", "content": test_case.prompt},
        ]
        response = call_model(
            self.target_client,
            self.target_model,
            messages,
            deployment=self.target_deployment,
            max_tokens=1024,
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

    def _evaluate_response(
        self, response: str, test_case: TestCase
    ) -> dict[str, float]:
        """Score a response on all metrics using LLM-as-Judge."""
        scores = {}
        for metric in self.metrics:
            result = _judge_response(
                self.judge_client,
                self.judge_deployment or self.judge_model,
                metric,
                response=response,
                query=test_case.prompt,
                context=test_case.context or "",
            )
            scores[metric] = float(result.get("score", 0))
        return scores

    def collect(self, test_cases: list | None = None, verbose: bool = True) -> tuple[list[dict], list[dict]]:
        """
        Call source and target models, return raw responses without scoring.

        Returns (source_items, target_items) — lists of dicts compatible
        with both SDK evaluate() and FoundryEvalsClient.evaluate_items().

        Each item: {"query": ..., "response": ..., "context": ..., "ground_truth": ...}
        """
        cases = test_cases or self.test_cases
        source_items = []
        target_items = []

        for i, tc in enumerate(cases):
            if verbose:
                prompt_preview = tc.prompt[:50].replace("\n", " ")
                _out(f"`[{i+1}/{len(cases)}]` Collecting: {prompt_preview}...")

            source_resp = self._call_source(tc)
            target_resp = self._call_target(tc)

            base = {
                "query": tc.prompt,
                "context": tc.context or "",
                "ground_truth": tc.expected_output or "",
            }
            source_items.append({**base, "response": source_resp})
            target_items.append({**base, "response": target_resp})

        return source_items, target_items

    def run(self, verbose: bool = True) -> ComparisonReport:
        """
        Run the full evaluation pipeline (legacy, uses built-in LLM-as-Judge).

        For most use cases, prefer collect() + quick_evaluate() or FoundryEvalsClient.

        For each test case:
        1. Call source model → get response
        2. Call target model → get response
        3. Judge both responses on all metrics
        4. Detect regressions

        Returns:
            ComparisonReport with all results.
        """
        results = []

        for i, tc in enumerate(self.test_cases):
            if verbose:
                prompt_preview = tc.prompt[:50].replace("\n", " ")
                print(f"  [{i+1}/{len(self.test_cases)}] Evaluating: {prompt_preview}...")

            # Get responses from both models
            source_resp = self._call_source(tc)
            target_resp = self._call_target(tc)

            # Score both with LLM-as-Judge
            source_scores = self._evaluate_response(source_resp, tc)
            target_scores = self._evaluate_response(target_resp, tc)

            # Detect regression: any metric drops below threshold
            regression = False
            for metric in self.metrics:
                delta = target_scores.get(metric, 0) - source_scores.get(metric, 0)
                if delta < self.regression_threshold:
                    regression = True
                    break

            results.append(
                EvalResult(
                    test_case=tc,
                    source_response=source_resp,
                    target_response=target_resp,
                    source_scores=source_scores,
                    target_scores=target_scores,
                    regression_detected=regression,
                )
            )

        return ComparisonReport(
            source_model=self.source_model,
            target_model=self.target_model,
            results=results,
            metrics=self.metrics,
            regression_threshold=self.regression_threshold,
        )

    def run_single(self, test_case: TestCase, verbose: bool = True) -> EvalResult:
        """Run evaluation for a single test case (useful for debugging)."""
        report = MigrationEvaluator(
            source_model=self.source_model,
            target_model=self.target_model,
            test_cases=[test_case],
            metrics=self.metrics,
            judge_model=self.judge_model,
            endpoint=None,
            api_key=None,
            source_deployment=self.source_deployment,
            target_deployment=self.target_deployment,
            judge_deployment=self.judge_deployment,
        ).run(verbose=verbose)
        return report.results[0]


# ---------------------------------------------------------------------------
# Utility: Load test cases from JSONL
# ---------------------------------------------------------------------------

def load_test_cases(path: str) -> list[TestCase]:
    """
    Load test cases from a JSONL file.

    Expected format (one JSON object per line):
    {"prompt": "...", "system_prompt": "...", "expected_output": "...", "context": "..."}

    Only "prompt" is required. Other fields are optional.
    """
    test_cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            test_cases.append(
                TestCase(
                    prompt=data["prompt"],
                    system_prompt=data.get("system_prompt", "You are a helpful assistant."),
                    expected_output=data.get("expected_output"),
                    context=data.get("context"),
                    tools=data.get("tools"),
                    ground_truth_label=data.get("ground_truth_label"),
                    metadata=data.get("metadata", {}),
                )
            )
    return test_cases
