"""
RAG (Retrieval-Augmented Generation) evaluation scenario.

Tests whether models can generate accurate, grounded responses based on
provided context documents. Key concern during migration: does the new model
hallucinate more or less than the old one?

Metrics:
- Groundedness: Is the response supported by the provided context?
- Relevance: Does the response address the user's question?
- Coherence: Is the response logically structured?
"""

from src.evaluate.core import TestCase, MigrationEvaluator
from src.evaluate.prompts import load_prompty


RAG_METRICS = ["groundedness", "relevance", "coherence"]

RAG_SYSTEM_PROMPT = load_prompty("rag")["system_prompt"]


# ---------------------------------------------------------------------------
# Sample test data: IT company policies (fictional)
# ---------------------------------------------------------------------------

RAG_TEST_CASES = [
    # 1. Simple factual extraction
    TestCase(
        prompt="What is the company's remote work policy?",
        system_prompt=RAG_SYSTEM_PROMPT,
        context="""## Remote Work Policy (Updated January 2025)

Employees are eligible for hybrid work after completing their 6-month probation period.
Hybrid employees must be present in the office a minimum of 2 days per week (Tuesday and Thursday are mandatory).
Fully remote positions require VP-level approval and are limited to roles in Engineering and Data Science.
All remote workers must be available during core hours: 9:00 AM - 3:00 PM CET.
Equipment allowance for remote setup: up to 1,500 EUR (one-time, requires manager approval).""",
        expected_output="Hybrid work after 6-month probation, minimum 2 days in office (Tuesday/Thursday mandatory), fully remote requires VP approval for Engineering/Data Science roles, core hours 9-3 CET, 1500 EUR equipment allowance.",
    ),
    # 2. Multi-fact question requiring synthesis
    TestCase(
        prompt="What are the conditions for getting a fully remote position and what equipment support is available?",
        system_prompt=RAG_SYSTEM_PROMPT,
        context="""## Remote Work Policy (Updated January 2025)

Employees are eligible for hybrid work after completing their 6-month probation period.
Hybrid employees must be present in the office a minimum of 2 days per week (Tuesday and Thursday are mandatory).
Fully remote positions require VP-level approval and are limited to roles in Engineering and Data Science.
All remote workers must be available during core hours: 9:00 AM - 3:00 PM CET.
Equipment allowance for remote setup: up to 1,500 EUR (one-time, requires manager approval).""",
        expected_output="Fully remote requires VP-level approval and is limited to Engineering and Data Science. Equipment allowance: up to 1,500 EUR one-time with manager approval.",
    ),
    # 3. Question NOT in context (tests hallucination resistance)
    TestCase(
        prompt="What is the company's parental leave policy?",
        system_prompt=RAG_SYSTEM_PROMPT,
        context="""## Remote Work Policy (Updated January 2025)

Employees are eligible for hybrid work after completing their 6-month probation period.
Hybrid employees must be present in the office a minimum of 2 days per week (Tuesday and Thursday are mandatory).
Fully remote positions require VP-level approval and are limited to roles in Engineering and Data Science.
All remote workers must be available during core hours: 9:00 AM - 3:00 PM CET.
Equipment allowance for remote setup: up to 1,500 EUR (one-time, requires manager approval).""",
        expected_output="I don't have enough information to answer this question.",
    ),
    # 4. Technical documentation RAG
    TestCase(
        prompt="How do I configure the API rate limiter for production?",
        system_prompt=RAG_SYSTEM_PROMPT,
        context="""## API Rate Limiting Configuration

The rate limiter uses a token bucket algorithm with the following default settings:
- Standard tier: 100 requests/minute, burst capacity of 150
- Premium tier: 500 requests/minute, burst capacity of 750
- Enterprise tier: Custom limits, contact support

Configuration file: `config/rate_limiter.yaml`

```yaml
rate_limiter:
  enabled: true
  algorithm: token_bucket
  default_tier: standard
  tiers:
    standard:
      requests_per_minute: 100
      burst_capacity: 150
    premium:
      requests_per_minute: 500
      burst_capacity: 750
```

For production deployments, set `RATE_LIMITER_ENV=production` in your environment variables.
This enables distributed rate limiting across multiple instances using Redis.
Redis connection is configured via `REDIS_URL` environment variable.

Important: Never disable rate limiting in production. Set minimum rate to 10 req/min.""",
        expected_output="Configure via config/rate_limiter.yaml, set RATE_LIMITER_ENV=production for distributed rate limiting with Redis (REDIS_URL env var). Standard: 100 req/min, Premium: 500 req/min.",
    ),
    # 5. Nuanced question with partial information
    TestCase(
        prompt="Can a junior developer in the marketing team work fully remote?",
        system_prompt=RAG_SYSTEM_PROMPT,
        context="""## Remote Work Policy (Updated January 2025)

Employees are eligible for hybrid work after completing their 6-month probation period.
Hybrid employees must be present in the office a minimum of 2 days per week (Tuesday and Thursday are mandatory).
Fully remote positions require VP-level approval and are limited to roles in Engineering and Data Science.
All remote workers must be available during core hours: 9:00 AM - 3:00 PM CET.
Equipment allowance for remote setup: up to 1,500 EUR (one-time, requires manager approval).""",
        expected_output="No, fully remote positions are limited to Engineering and Data Science roles. A marketing team member would not be eligible for fully remote work under the current policy.",
    ),
    # 6. Financial report RAG
    TestCase(
        prompt="What was the revenue growth compared to the previous year?",
        system_prompt=RAG_SYSTEM_PROMPT,
        context="""## Q3 2025 Financial Summary

Total Revenue: EUR 45.2M (Q3 2024: EUR 38.7M)
Operating Income: EUR 8.1M (Q3 2024: EUR 6.9M)
Net Profit Margin: 17.9% (Q3 2024: 17.8%)
Employee count: 1,247 (up from 1,102)

Key highlights:
- SaaS revenue grew 22% YoY to EUR 31.4M
- Professional services revenue declined 3% to EUR 13.8M
- Customer acquisition cost decreased by 15%
- Net Revenue Retention Rate: 118%""",
        expected_output="Total revenue grew from EUR 38.7M to EUR 45.2M, representing approximately 16.8% year-over-year growth. SaaS revenue specifically grew 22% YoY.",
    ),
    # 7. Legal/compliance context
    TestCase(
        prompt="What data can we store about EU customers and for how long?",
        system_prompt=RAG_SYSTEM_PROMPT,
        context="""## Data Retention Policy - EU Customers (GDPR Compliant)

Personal data categories and retention periods:
- Account information (name, email, company): Duration of contract + 2 years
- Transaction history: 7 years (legal requirement for financial records)
- Usage analytics: 18 months (anonymized after 6 months)
- Support tickets: 3 years from resolution date
- Marketing consent records: Duration of consent + 1 year after withdrawal

Data must be stored in EU data centers (primary: Frankfurt, backup: Amsterdam).
Cross-border transfers require Standard Contractual Clauses (SCCs).
Data Subject Access Requests (DSARs) must be fulfilled within 30 days.

Prohibited: Storing biometric data, political opinions, health data without explicit consent and DPO approval.""",
        expected_output="Account info: contract duration + 2 years. Transactions: 7 years. Usage analytics: 18 months (anonymized after 6). Support tickets: 3 years. Marketing consent: consent duration + 1 year. Must be stored in EU (Frankfurt/Amsterdam). No biometric/political/health data without explicit consent and DPO approval.",
    ),
    # 8. Product documentation with edge case
    TestCase(
        prompt="What happens if a webhook delivery fails?",
        system_prompt=RAG_SYSTEM_PROMPT,
        context="""## Webhook Configuration Guide

Webhooks are sent as HTTP POST requests with JSON payload.
Timeout: 10 seconds. If the endpoint does not respond within 10 seconds, the delivery is marked as failed.

Retry policy:
- First retry: 1 minute after failure
- Second retry: 5 minutes after first retry
- Third retry: 30 minutes after second retry
- After 3 failed retries, the webhook is marked as permanently failed

Failed webhook notifications are stored in the dead letter queue for 72 hours.
Administrators can manually replay failed webhooks from the dashboard.
If an endpoint fails consistently (>90% failure rate over 24 hours), it is automatically disabled.

Webhook signature: Each request includes an X-Webhook-Signature header using HMAC-SHA256.""",
        expected_output="Failed deliveries are retried 3 times (1 min, 5 min, 30 min intervals). After 3 failures, moved to dead letter queue (72 hours). Can be manually replayed. Endpoints with >90% failure rate over 24h are auto-disabled.",
    ),
]


def create_rag_evaluator(
    source_model: str = "gpt-4o",
    target_model: str = "gpt-4.1",
    test_cases: list[TestCase] | None = None,
    **kwargs,
) -> MigrationEvaluator:
    """
    Create a pre-configured evaluator for RAG scenarios.

    Args:
        source_model: Current model to migrate from.
        target_model: New model to migrate to.
        test_cases: Custom test cases (uses built-in examples if None).
        **kwargs: Additional arguments passed to MigrationEvaluator.

    Returns:
        Configured MigrationEvaluator ready to run.

    Example:
        evaluator = create_rag_evaluator("gpt-4o", "gpt-4.1")
        report = evaluator.run()
        report.print_report()
    """
    return MigrationEvaluator(
        source_model=source_model,
        target_model=target_model,
        test_cases=test_cases or RAG_TEST_CASES,
        metrics=RAG_METRICS,
        **kwargs,
    )
