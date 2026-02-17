"""
Classification / Sentiment Analysis evaluation scenario.

Tests whether models maintain classification accuracy across migration.
Provides labeled examples with ground truth for deterministic evaluation.

Metrics:
- accuracy: Does the model return the correct label?
- consistency: Does the model return the same label across multiple runs?
- relevance: Is the response well-formatted and on-topic?
"""

import json
from typing import Any

from src.evaluate.core import (
    TestCase,
    MigrationEvaluator,
    ComparisonReport,
    EvalResult,
    _judge_response,
)
from src.clients import create_client, call_model
from src.config import MODEL_REGISTRY
from src.evaluate.core import _out
from src.evaluate.prompts import load_prompty


CLASSIFICATION_METRICS = ["accuracy", "consistency", "relevance"]


# ---------------------------------------------------------------------------
# System prompts loaded from .prompty files
# ---------------------------------------------------------------------------

SENTIMENT_SYSTEM_PROMPT = load_prompty("classify_sentiment")["system_prompt"]
CATEGORY_SYSTEM_PROMPT = load_prompty("classify_category")["system_prompt"]
INTENT_SYSTEM_PROMPT = load_prompty("classify_intent")["system_prompt"]
PRIORITY_SYSTEM_PROMPT = load_prompty("classify_priority")["system_prompt"]


# ---------------------------------------------------------------------------
# Sample test cases with ground truth labels
# ---------------------------------------------------------------------------

CLASSIFICATION_TEST_CASES = [
    # ── Sentiment Analysis ────────────────────────────────────────────────
    TestCase(
        prompt="This product exceeded all my expectations! The build quality is outstanding and the customer service team was incredibly helpful when I had questions. Highly recommend!",
        system_prompt=SENTIMENT_SYSTEM_PROMPT,
        ground_truth_label="POSITIVE",
        metadata={"task": "sentiment", "difficulty": "easy"},
    ),
    TestCase(
        prompt="Absolutely terrible experience. The item arrived broken, customer support took 3 weeks to respond, and they refused to issue a refund. Never buying from this company again.",
        system_prompt=SENTIMENT_SYSTEM_PROMPT,
        ground_truth_label="NEGATIVE",
        metadata={"task": "sentiment", "difficulty": "easy"},
    ),
    TestCase(
        prompt="The package arrived on time. The product works as described in the specifications. Nothing particularly special or disappointing about it.",
        system_prompt=SENTIMENT_SYSTEM_PROMPT,
        ground_truth_label="NEUTRAL",
        metadata={"task": "sentiment", "difficulty": "easy"},
    ),
    # Nuanced / mixed sentiment
    TestCase(
        prompt="The food was absolutely delicious but the service was painfully slow. We waited 45 minutes for our main course. I'd come back for the food but definitely not during peak hours.",
        system_prompt=SENTIMENT_SYSTEM_PROMPT,
        ground_truth_label="NEUTRAL",
        metadata={"task": "sentiment", "difficulty": "hard"},
    ),
    # Sarcasm detection
    TestCase(
        prompt="Oh great, another software update that breaks everything. Just what I needed on a Friday afternoon. Thanks so much for making my weekend productive fixing your bugs.",
        system_prompt=SENTIMENT_SYSTEM_PROMPT,
        ground_truth_label="NEGATIVE",
        metadata={"task": "sentiment", "difficulty": "hard"},
    ),

    # ── Support Ticket Classification ─────────────────────────────────────
    TestCase(
        prompt="I was charged twice for my subscription this month. My credit card shows two identical charges of $29.99 on March 1st. Please refund the duplicate charge.",
        system_prompt=CATEGORY_SYSTEM_PROMPT,
        ground_truth_label="BILLING",
        metadata={"task": "support_ticket", "difficulty": "easy"},
    ),
    TestCase(
        prompt="The app keeps crashing whenever I try to upload a file larger than 10MB. I'm running version 3.2.1 on iOS 18. I've already tried reinstalling the app.",
        system_prompt=CATEGORY_SYSTEM_PROMPT,
        ground_truth_label="TECHNICAL",
        metadata={"task": "support_ticket", "difficulty": "easy"},
    ),
    TestCase(
        prompt="I want to change the email address associated with my account. My old email john@oldcompany.com is no longer active and I need to switch to john@newcompany.com.",
        system_prompt=CATEGORY_SYSTEM_PROMPT,
        ground_truth_label="ACCOUNT",
        metadata={"task": "support_ticket", "difficulty": "easy"},
    ),
    TestCase(
        prompt="My order #12345 shows as delivered but I never received it. The tracking says it was left at the front door but nothing was there. I have a Ring camera and no delivery was captured.",
        system_prompt=CATEGORY_SYSTEM_PROMPT,
        ground_truth_label="SHIPPING",
        metadata={"task": "support_ticket", "difficulty": "easy"},
    ),
    # Ambiguous ticket (could be TECHNICAL or PRODUCT)
    TestCase(
        prompt="The battery on my wireless headphones only lasts 2 hours instead of the advertised 8 hours. Is this a defect or is there a firmware update that fixes this?",
        system_prompt=CATEGORY_SYSTEM_PROMPT,
        ground_truth_label="PRODUCT",
        metadata={"task": "support_ticket", "difficulty": "hard"},
    ),

    # ── Intent Classification ─────────────────────────────────────────────
    TestCase(
        prompt="I need to fly from Paris to Tokyo next Thursday, returning on Sunday. Business class for 2 passengers.",
        system_prompt=INTENT_SYSTEM_PROMPT,
        ground_truth_label="BOOK_FLIGHT",
        metadata={"task": "intent", "difficulty": "easy"},
    ),
    TestCase(
        prompt="Something came up and I can no longer travel on March 15th. I need to cancel my booking reference ABC123 and get a refund.",
        system_prompt=INTENT_SYSTEM_PROMPT,
        ground_truth_label="CANCEL_BOOKING",
        metadata={"task": "intent", "difficulty": "easy"},
    ),
    TestCase(
        prompt="Can I bring my guitar as carry-on luggage? It's in a hard case and measures about 105cm. Also, what's the weight limit for checked bags on international flights?",
        system_prompt=INTENT_SYSTEM_PROMPT,
        ground_truth_label="BAGGAGE_INFO",
        metadata={"task": "intent", "difficulty": "standard"},
    ),

    # ── IT Incident Priority ──────────────────────────────────────────────
    TestCase(
        prompt="URGENT: Production database is not responding. All customer-facing services are returning 500 errors. Approximately 50,000 users affected. Started 5 minutes ago.",
        system_prompt=PRIORITY_SYSTEM_PROMPT,
        ground_truth_label="P1_CRITICAL",
        metadata={"task": "priority", "difficulty": "easy"},
    ),
    TestCase(
        prompt="The export to PDF feature is generating files with misaligned tables. Users can still export to CSV as a workaround. Affects the monthly reporting module.",
        system_prompt=PRIORITY_SYSTEM_PROMPT,
        ground_truth_label="P3_MEDIUM",
        metadata={"task": "priority", "difficulty": "standard"},
    ),
    TestCase(
        prompt="It would be nice if the dashboard had a dark mode option. Several users have requested this feature. Not urgent but would improve the user experience.",
        system_prompt=PRIORITY_SYSTEM_PROMPT,
        ground_truth_label="P4_LOW",
        metadata={"task": "priority", "difficulty": "easy"},
    ),
]


# ---------------------------------------------------------------------------
# Custom classification evaluator
# ---------------------------------------------------------------------------

class ClassificationEvaluator(MigrationEvaluator):
    """
    Evaluator specialized for classification tasks.

    Compares the model's output label against ground truth and measures
    consistency across multiple runs.
    """

    def __init__(self, consistency_runs: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.consistency_runs = consistency_runs

    def collect(self, test_cases: list | None = None, verbose: bool = True) -> tuple[list[dict], list[dict]]:
        """
        Classify with source and target models, return raw predictions.

        Returns (source_items, target_items) with predictions and
        deterministic accuracy scores, compatible with FoundryEvalsClient.
        """
        cases = test_cases or self.test_cases
        source_items = []
        target_items = []

        for i, tc in enumerate(cases):
            if verbose:
                task = tc.metadata.get("task", "classification")
                label = tc.ground_truth_label or "?"
                _out(f"`[{i+1}/{len(cases)}]` Collecting: {task} - expected: {label}")

            source_pred = self._classify(
                self.source_client, self.source_model, tc, self.source_deployment
            )
            target_pred = self._classify(
                self.target_client, self.target_model, tc, self.target_deployment
            )

            base = {
                "query": tc.prompt,
                "context": "",
                "ground_truth": tc.ground_truth_label or "",
            }

            source_accuracy = self._check_accuracy(source_pred, tc.ground_truth_label) if tc.ground_truth_label else 3.0
            target_accuracy = self._check_accuracy(target_pred, tc.ground_truth_label) if tc.ground_truth_label else 3.0

            source_items.append({
                **base,
                "response": source_pred,
                "_prediction": source_pred,
                "_accuracy": source_accuracy,
            })
            target_items.append({
                **base,
                "response": target_pred,
                "_prediction": target_pred,
                "_accuracy": target_accuracy,
            })

        return source_items, target_items

    def _classify(self, client, model_name: str, tc: TestCase, deployment: str | None = None) -> str:
        """Get classification label from model."""
        response = call_model(
            client,
            model_name,
            messages=[
                {"role": "system", "content": tc.system_prompt},
                {"role": "user", "content": tc.prompt},
            ],
            deployment=deployment,
            max_tokens=50,
            temperature=0.0,
        )
        return (response.choices[0].message.content or "").strip().upper()

    def _check_accuracy(self, predicted: str, expected: str) -> float:
        """Check if prediction matches ground truth (5.0 = correct, 1.0 = wrong)."""
        # Normalize: remove extra whitespace, underscores vs spaces
        predicted_clean = predicted.strip().replace(" ", "_").upper()
        expected_clean = expected.strip().replace(" ", "_").upper()

        if predicted_clean == expected_clean:
            return 5.0
        # Partial match (e.g., "POSITIVE" in "POSITIVE - the text is clearly positive")
        if expected_clean in predicted_clean:
            return 4.0
        return 1.0

    def _check_consistency(self, client, model_name: str, tc: TestCase, deployment: str | None) -> float:
        """Run classification multiple times and check consistency (5.0 = always same, 1.0 = all different)."""
        labels = []
        for _ in range(self.consistency_runs):
            label = self._classify(client, model_name, tc, deployment)
            labels.append(label)

        # Most common label
        from collections import Counter
        counter = Counter(labels)
        most_common_count = counter.most_common(1)[0][1]
        consistency_ratio = most_common_count / len(labels)

        return round(1.0 + 4.0 * consistency_ratio, 1)

    def run(self, verbose: bool = True) -> ComparisonReport:
        """Run classification evaluation with accuracy and consistency checks."""
        results = []

        for i, tc in enumerate(self.test_cases):
            if verbose:
                task = tc.metadata.get("task", "classification")
                diff = tc.metadata.get("difficulty", "")
                label = tc.ground_truth_label or "?"
                print(f"  [{i+1}/{len(self.test_cases)}] {task} ({diff}) - expected: {label}")

            # Get predictions
            source_pred = self._classify(
                self.source_client, self.source_model, tc, self.source_deployment
            )
            target_pred = self._classify(
                self.target_client, self.target_model, tc, self.target_deployment
            )

            # Score accuracy
            source_scores: dict[str, float] = {}
            target_scores: dict[str, float] = {}

            if tc.ground_truth_label:
                source_scores["accuracy"] = self._check_accuracy(source_pred, tc.ground_truth_label)
                target_scores["accuracy"] = self._check_accuracy(target_pred, tc.ground_truth_label)
            else:
                source_scores["accuracy"] = 3.0
                target_scores["accuracy"] = 3.0

            # Check consistency
            source_scores["consistency"] = self._check_consistency(
                self.source_client, self.source_model, tc, self.source_deployment
            )
            target_scores["consistency"] = self._check_consistency(
                self.target_client, self.target_model, tc, self.target_deployment
            )

            # Score relevance of response format via LLM-as-Judge
            source_relevance = _judge_response(
                self.judge_client,
                self.judge_deployment or self.judge_model,
                "relevance",
                response=source_pred,
                query=f"Classify: {tc.prompt[:100]}",
            )
            target_relevance = _judge_response(
                self.judge_client,
                self.judge_deployment or self.judge_model,
                "relevance",
                response=target_pred,
                query=f"Classify: {tc.prompt[:100]}",
            )
            source_scores["relevance"] = float(source_relevance.get("score", 0))
            target_scores["relevance"] = float(target_relevance.get("score", 0))

            # Detect regression
            regression = False
            for metric in self.metrics:
                delta = target_scores.get(metric, 0) - source_scores.get(metric, 0)
                if delta < self.regression_threshold:
                    regression = True
                    break

            results.append(
                EvalResult(
                    test_case=tc,
                    source_response=f"{source_pred} (expected: {tc.ground_truth_label})",
                    target_response=f"{target_pred} (expected: {tc.ground_truth_label})",
                    source_scores=source_scores,
                    target_scores=target_scores,
                    regression_detected=regression,
                    details={
                        "source_prediction": source_pred,
                        "target_prediction": target_pred,
                        "ground_truth": tc.ground_truth_label,
                    },
                )
            )

        return ComparisonReport(
            source_model=self.source_model,
            target_model=self.target_model,
            results=results,
            metrics=self.metrics,
            regression_threshold=self.regression_threshold,
        )


def create_classification_evaluator(
    source_model: str = "gpt-4o",
    target_model: str = "gpt-4.1",
    test_cases: list[TestCase] | None = None,
    consistency_runs: int = 3,
    **kwargs,
) -> ClassificationEvaluator:
    """
    Create a pre-configured evaluator for classification / sentiment analysis.

    Args:
        source_model: Current model to migrate from.
        target_model: New model to migrate to.
        test_cases: Custom test cases (uses built-in examples if None).
        consistency_runs: Number of times to run each test for consistency check.

    Returns:
        Configured ClassificationEvaluator ready to run.

    Example:
        evaluator = create_classification_evaluator("gpt-4o", "gpt-4.1")
        report = evaluator.run()
        report.print_report()
    """
    return ClassificationEvaluator(
        source_model=source_model,
        target_model=target_model,
        test_cases=test_cases or CLASSIFICATION_TEST_CASES,
        metrics=CLASSIFICATION_METRICS,
        consistency_runs=consistency_runs,
        **kwargs,
    )
