"""
Tool Calling evaluation scenario.

Tests whether models correctly select tools and extract parameters from
user queries. Key concern during migration: does the new model call the
right tools with the right arguments?

Metrics:
- tool_accuracy: Did the model select the correct tool?
- param_accuracy: Did the model extract the correct parameters?
- relevance: Is the final response relevant to the query?
"""

import json
from typing import Any, Optional

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


TOOL_CALLING_METRICS = ["tool_accuracy", "param_accuracy", "relevance"]


# ---------------------------------------------------------------------------
# Sample tools definition (fictional multi-service assistant)
# ---------------------------------------------------------------------------

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather and forecast for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "country": {"type": "string", "description": "Country code (ISO 3166-1 alpha-2)"},
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of forecast days (1-7)",
                        "default": 1,
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search for products in the catalog by name, category, or price range",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "category": {
                        "type": "string",
                        "enum": ["electronics", "clothing", "home", "sports", "books"],
                        "description": "Product category filter",
                    },
                    "min_price": {"type": "number", "description": "Minimum price in EUR"},
                    "max_price": {"type": "number", "description": "Maximum price in EUR"},
                    "sort_by": {
                        "type": "string",
                        "enum": ["price_asc", "price_desc", "rating", "relevance"],
                        "default": "relevance",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Create a new calendar event with title, date, time, and optional attendees",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Event title"},
                    "date": {"type": "string", "description": "Event date (YYYY-MM-DD)"},
                    "start_time": {"type": "string", "description": "Start time (HH:MM)"},
                    "end_time": {"type": "string", "description": "End time (HH:MM)"},
                    "attendees": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of attendee email addresses",
                    },
                    "location": {"type": "string", "description": "Meeting location or URL"},
                    "reminder_minutes": {
                        "type": "integer",
                        "description": "Reminder before event in minutes",
                        "default": 15,
                    },
                },
                "required": ["title", "date", "start_time", "end_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to one or more recipients",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Recipient email addresses",
                    },
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body (plain text or HTML)"},
                    "cc": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "CC recipients",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "normal", "high"],
                        "default": "normal",
                    },
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_info",
            "description": "Get information about product inventory and stock levels",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string", "description": "Product name to check"},
                    "warehouse": {
                        "type": "string",
                        "enum": ["paris", "london", "berlin"],
                        "description": "Warehouse location",
                    },
                },
                "required": ["product_name"],
            },
        },
    },
]

TOOL_CALLING_SYSTEM_PROMPT = load_prompty("tool_calling")["system_prompt"]


# ---------------------------------------------------------------------------
# Sample test cases with expected tool calls
# ---------------------------------------------------------------------------

TOOL_CALLING_TEST_CASES = [
    # 1. Simple single-tool call
    TestCase(
        prompt="What's the weather like in Paris today?",
        system_prompt=TOOL_CALLING_SYSTEM_PROMPT,
        tools=SAMPLE_TOOLS,
        metadata={
            "expected_tool": "get_weather",
            "expected_params": {"city": "Paris", "country": "FR"},
            "description": "Simple weather query - should call get_weather with city=Paris",
        },
    ),
    # 2. Tool with specific parameters
    TestCase(
        prompt="Show me running shoes under 100 euros, sorted by best rating",
        system_prompt=TOOL_CALLING_SYSTEM_PROMPT,
        tools=SAMPLE_TOOLS,
        metadata={
            "expected_tool": "search_products",
            "expected_params": {
                "query": "running shoes",
                "category": "sports",
                "max_price": 100,
                "sort_by": "rating",
            },
            "description": "Product search with category, price, and sort filters",
        },
    ),
    # 3. Calendar event with multiple parameters
    TestCase(
        prompt="Schedule a team standup meeting for next Monday March 3rd from 9:00 to 9:30 AM with alice@company.com and bob@company.com in the main conference room",
        system_prompt=TOOL_CALLING_SYSTEM_PROMPT,
        tools=SAMPLE_TOOLS,
        metadata={
            "expected_tool": "create_calendar_event",
            "expected_params": {
                "title": "Team Standup",
                "date": "2026-03-03",
                "start_time": "09:00",
                "end_time": "09:30",
                "attendees": ["alice@company.com", "bob@company.com"],
                "location": "Main Conference Room",
            },
            "description": "Calendar event with attendees and location extraction",
        },
    ),
    # 4. Email composition
    TestCase(
        prompt="Send an email to alice@company.com with subject 'Team Lunch Friday' saying we'll meet at noon at the Italian restaurant on Main Street",
        system_prompt=TOOL_CALLING_SYSTEM_PROMPT,
        tools=SAMPLE_TOOLS,
        metadata={
            "expected_tool": "send_email",
            "expected_params": {
                "to": ["alice@company.com"],
                "subject": "Team Lunch Friday",
            },
            "description": "Simple email composition with recipient and subject extraction",
        },
    ),
    # 5. Stock check
    TestCase(
        prompt="Do we have any wireless keyboards in the Paris warehouse?",
        system_prompt=TOOL_CALLING_SYSTEM_PROMPT,
        tools=SAMPLE_TOOLS,
        metadata={
            "expected_tool": "get_stock_info",
            "expected_params": {
                "product_name": "wireless keyboard",
                "warehouse": "paris",
            },
            "description": "Stock check with product name and warehouse extraction",
        },
    ),
    # 6. Ambiguous request - should pick the right tool
    TestCase(
        prompt="I need to know about laptop deals",
        system_prompt=TOOL_CALLING_SYSTEM_PROMPT,
        tools=SAMPLE_TOOLS,
        metadata={
            "expected_tool": "search_products",
            "expected_params": {
                "query": "laptop",
                "category": "electronics",
            },
            "description": "Ambiguous request - should map to product search with electronics category",
        },
    ),
    # 7. Weather with units specification
    TestCase(
        prompt="What will the temperature be in New York for the next 3 days? Give me Fahrenheit please.",
        system_prompt=TOOL_CALLING_SYSTEM_PROMPT,
        tools=SAMPLE_TOOLS,
        metadata={
            "expected_tool": "get_weather",
            "expected_params": {
                "city": "New York",
                "units": "fahrenheit",
                "days": 3,
            },
            "description": "Weather with explicit unit and multi-day forecast",
        },
    ),
    # 8. No tool needed - direct response
    TestCase(
        prompt="What is 25 * 4?",
        system_prompt=TOOL_CALLING_SYSTEM_PROMPT,
        tools=SAMPLE_TOOLS,
        metadata={
            "expected_tool": None,
            "expected_params": {},
            "description": "Simple math - should respond directly without calling a tool",
        },
    ),
]


# ---------------------------------------------------------------------------
# Custom tool-calling evaluator
# ---------------------------------------------------------------------------

class ToolCallingEvaluator(MigrationEvaluator):
    """
    Evaluator specialized for tool-calling comparison.

    Instead of only using LLM-as-Judge for text quality, this evaluator also
    checks whether the correct tool was selected and parameters were extracted.
    """

    def collect(self, test_cases: list | None = None, verbose: bool = True) -> tuple[list[dict], list[dict]]:
        """
        Call source and target models with tools, return raw results.

        Returns (source_items, target_items) with tool call details,
        compatible with FoundryEvalsClient.evaluate_items().
        """
        cases = test_cases or self.test_cases
        source_items = []
        target_items = []

        for i, tc in enumerate(cases):
            if verbose:
                desc = tc.metadata.get("description", tc.prompt[:50])
                _out(f"`[{i+1}/{len(cases)}]` Collecting: {desc}")

            source_result = self._call_with_tools(
                self.source_client, self.source_model, tc, self.source_deployment
            )
            target_result = self._call_with_tools(
                self.target_client, self.target_model, tc, self.target_deployment
            )

            base = {
                "query": tc.prompt,
                "context": "",
                "tool_definitions": json.dumps(tc.tools or []),
            }

            source_text = source_result["content"] or json.dumps(source_result["tool_calls"])
            target_text = target_result["content"] or json.dumps(target_result["tool_calls"])

            source_items.append({
                **base,
                "response": source_text,
                "tool_calls": json.dumps(source_result["tool_calls"]),
                "_raw": source_result,
                "_scores": self._score_tool_call(source_result, tc),
            })
            target_items.append({
                **base,
                "response": target_text,
                "tool_calls": json.dumps(target_result["tool_calls"]),
                "_raw": target_result,
                "_scores": self._score_tool_call(target_result, tc),
            })

        return source_items, target_items

    def _call_with_tools(self, client, model_name: str, tc: TestCase, deployment: str | None = None) -> dict:
        """Call model with tools and return tool call details."""
        messages = [
            {"role": "system", "content": tc.system_prompt},
            {"role": "user", "content": tc.prompt},
        ]

        model_config = MODEL_REGISTRY.get(model_name)
        params: dict[str, Any] = {}
        if model_config and model_config.max_tokens_param == "max_completion_tokens":
            params["max_completion_tokens"] = 1024
        else:
            params["max_tokens"] = 1024

        if model_config and model_config.supports_temperature:
            params["temperature"] = 0.0

        response = client.chat.completions.create(
            model=deployment or model_name,
            messages=messages,
            tools=tc.tools,
            **params,
        )

        choice = response.choices[0]
        result = {
            "content": choice.message.content or "",
            "tool_calls": [],
            "finish_reason": choice.finish_reason,
        }

        if choice.message.tool_calls:
            for tc_call in choice.message.tool_calls:
                try:
                    args = json.loads(tc_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                result["tool_calls"].append({
                    "name": tc_call.function.name,
                    "arguments": args,
                })

        return result

    def _score_tool_call(self, result: dict, test_case: TestCase) -> dict[str, float]:
        """Score tool call accuracy against expected values."""
        expected_tool = test_case.metadata.get("expected_tool")
        expected_params = test_case.metadata.get("expected_params", {})
        scores: dict[str, float] = {}

        # Tool accuracy: did it call the right tool (or no tool when expected)?
        if expected_tool is None:
            # Should NOT have called a tool
            scores["tool_accuracy"] = 5.0 if not result["tool_calls"] else 1.0
        elif result["tool_calls"]:
            actual_tool = result["tool_calls"][0]["name"]
            scores["tool_accuracy"] = 5.0 if actual_tool == expected_tool else 1.0
        else:
            scores["tool_accuracy"] = 1.0  # Expected a tool call but got none

        # Parameter accuracy: check key expected params are present and correct
        if expected_tool and result["tool_calls"]:
            actual_args = result["tool_calls"][0].get("arguments", {})
            if not expected_params:
                scores["param_accuracy"] = 5.0
            else:
                matched = 0
                total = len(expected_params)
                for key, expected_val in expected_params.items():
                    actual_val = actual_args.get(key)
                    if actual_val is not None:
                        # Flexible comparison
                        if isinstance(expected_val, str):
                            if expected_val.lower() in str(actual_val).lower():
                                matched += 1
                        elif isinstance(expected_val, (int, float)):
                            if actual_val == expected_val:
                                matched += 1
                        elif isinstance(expected_val, list):
                            if set(str(v).lower() for v in expected_val) <= set(str(v).lower() for v in actual_val):
                                matched += 1
                        else:
                            if actual_val == expected_val:
                                matched += 1
                scores["param_accuracy"] = round(1.0 + 4.0 * (matched / max(total, 1)), 1)
        elif expected_tool is None:
            scores["param_accuracy"] = 5.0  # N/A, no tool expected
        else:
            scores["param_accuracy"] = 1.0

        return scores

    def run(self, verbose: bool = True) -> ComparisonReport:
        """Run tool-calling evaluation with tool-specific scoring."""
        results = []

        for i, tc in enumerate(self.test_cases):
            if verbose:
                desc = tc.metadata.get("description", tc.prompt[:50])
                print(f"  [{i+1}/{len(self.test_cases)}] {desc}")

            # Call both models with tools
            source_result = self._call_with_tools(
                self.source_client, self.source_model, tc, self.source_deployment
            )
            target_result = self._call_with_tools(
                self.target_client, self.target_model, tc, self.target_deployment
            )

            # Score tool accuracy (deterministic)
            source_scores = self._score_tool_call(source_result, tc)
            target_scores = self._score_tool_call(target_result, tc)

            # Also score relevance of any text response via LLM-as-Judge
            source_text = source_result["content"] or json.dumps(source_result["tool_calls"])
            target_text = target_result["content"] or json.dumps(target_result["tool_calls"])

            source_relevance = _judge_response(
                self.judge_client,
                self.judge_deployment or self.judge_model,
                "relevance",
                response=source_text,
                query=tc.prompt,
            )
            target_relevance = _judge_response(
                self.judge_client,
                self.judge_deployment or self.judge_model,
                "relevance",
                response=target_text,
                query=tc.prompt,
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
                    source_response=json.dumps(source_result, indent=2, ensure_ascii=False),
                    target_response=json.dumps(target_result, indent=2, ensure_ascii=False),
                    source_scores=source_scores,
                    target_scores=target_scores,
                    regression_detected=regression,
                    details={
                        "source_tool_calls": source_result["tool_calls"],
                        "target_tool_calls": target_result["tool_calls"],
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


def create_tool_calling_evaluator(
    source_model: str = "gpt-4o",
    target_model: str = "gpt-4.1",
    test_cases: list[TestCase] | None = None,
    **kwargs,
) -> ToolCallingEvaluator:
    """
    Create a pre-configured evaluator for tool-calling scenarios.

    Args:
        source_model: Current model to migrate from.
        target_model: New model to migrate to.
        test_cases: Custom test cases (uses built-in examples if None).

    Returns:
        Configured ToolCallingEvaluator ready to run.

    Example:
        evaluator = create_tool_calling_evaluator("gpt-4o", "gpt-4.1")
        report = evaluator.run()
        report.print_report()
    """
    return ToolCallingEvaluator(
        source_model=source_model,
        target_model=target_model,
        test_cases=test_cases or TOOL_CALLING_TEST_CASES,
        metrics=TOOL_CALLING_METRICS,
        **kwargs,
    )
