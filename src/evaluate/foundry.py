"""
Azure AI Foundry Evals API client for migration evaluation.

Uses the Foundry Evals API (client.evals) to run cloud-based evaluations
with built-in evaluators. Results are visible in the Foundry portal.

Based on the proven FoundryEvalsClient pattern from workflow_metrics.

Supports metrics:
- Text quality: coherence, fluency, relevance, groundedness
- Agent quality: intent_resolution, task_adherence, task_completion, response_completeness
- Tool usage: tool_call_accuracy, tool_input_accuracy, tool_output_utilization,
              tool_selection, tool_success
"""

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from src.evaluate.core import ComparisonReport, _out


# ---------------------------------------------------------------------------
# Metric configurations (from Azure AI Foundry documentation)
# ---------------------------------------------------------------------------

FOUNDRY_METRICS = {
    # Text quality (score 1-5)
    "coherence": {
        "evaluator_name": "builtin.coherence",
        "data_mapping": {
            "query": "{{item.query}}",
            "response": "{{item.response}}",
        },
    },
    "fluency": {
        "evaluator_name": "builtin.fluency",
        "data_mapping": {
            "query": "{{item.query}}",
            "response": "{{item.response}}",
        },
    },
    "relevance": {
        "evaluator_name": "builtin.relevance",
        "data_mapping": {
            "query": "{{item.query}}",
            "response": "{{item.response}}",
        },
    },
    "groundedness": {
        "evaluator_name": "builtin.groundedness",
        "data_mapping": {
            "context": "{{item.context}}",
            "query": "{{item.query}}",
            "response": "{{item.response}}",
        },
    },
    # Agent quality
    "intent_resolution": {
        "evaluator_name": "builtin.intent_resolution",
        "data_mapping": {
            "query": "{{item.query}}",
            "response": "{{item.response}}",
            "tool_definitions": "{{item.tool_definitions}}",
        },
    },
    "task_adherence": {
        "evaluator_name": "builtin.task_adherence",
        "data_mapping": {
            "query": "{{item.query}}",
            "response": "{{item.response}}",
            "tool_definitions": "{{item.tool_definitions}}",
        },
    },
    "task_completion": {
        "evaluator_name": "builtin.task_completion",
        "data_mapping": {
            "query": "{{item.query}}",
            "response": "{{item.response}}",
            "tool_definitions": "{{item.tool_definitions}}",
        },
    },
    "response_completeness": {
        "evaluator_name": "builtin.response_completeness",
        "data_mapping": {
            "ground_truth": "{{item.ground_truth}}",
            "response": "{{item.response}}",
        },
    },
    # Tool usage
    "tool_call_accuracy": {
        "evaluator_name": "builtin.tool_call_accuracy",
        "data_mapping": {
            "query": "{{item.query}}",
            "tool_definitions": "{{item.tool_definitions}}",
            "tool_calls": "{{item.tool_calls}}",
            "response": "{{item.response}}",
        },
    },
    "tool_input_accuracy": {
        "evaluator_name": "builtin.tool_input_accuracy",
        "data_mapping": {
            "query": "{{item.query}}",
            "response": "{{item.response}}",
            "tool_definitions": "{{item.tool_definitions}}",
        },
    },
    "tool_output_utilization": {
        "evaluator_name": "builtin.tool_output_utilization",
        "data_mapping": {
            "query": "{{item.query}}",
            "response": "{{item.response}}",
            "tool_definitions": "{{item.tool_definitions}}",
        },
    },
    "tool_selection": {
        "evaluator_name": "builtin.tool_selection",
        "data_mapping": {
            "query": "{{item.query}}",
            "response": "{{item.response}}",
            "tool_calls": "{{item.tool_calls}}",
            "tool_definitions": "{{item.tool_definitions}}",
        },
    },
    "tool_success": {
        "evaluator_name": "builtin.tool_call_success",
        "data_mapping": {
            "tool_definitions": "{{item.tool_definitions}}",
            "response": "{{item.response}}",
        },
    },
}

TEXT_QUALITY_METRICS = ["coherence", "fluency", "relevance", "groundedness"]
AGENT_QUALITY_METRICS = ["intent_resolution", "task_adherence", "task_completion", "response_completeness"]
TOOL_USAGE_METRICS = ["tool_call_accuracy", "tool_input_accuracy", "tool_output_utilization", "tool_selection", "tool_success"]
ALL_FOUNDRY_METRICS = TEXT_QUALITY_METRICS + AGENT_QUALITY_METRICS + TOOL_USAGE_METRICS


# ---------------------------------------------------------------------------
# Foundry Evals Client
# ---------------------------------------------------------------------------

class FoundryEvalsClient:
    """
    Azure Foundry Evals API client.

    Uses client.evals.create() + client.evals.runs.create() to submit
    evaluations, then polls for completion and parses results.

    Results are visible in the Azure AI Foundry portal.

    Usage:
        client = FoundryEvalsClient()

        # Evaluate a ComparisonReport (migration A/B testing)
        results = client.evaluate_comparison(report, metrics=["coherence", "fluency", "relevance"])

        # Or evaluate raw items directly
        results = client.evaluate_items(items, metrics=["coherence", "fluency"])
    """

    def __init__(
        self,
        endpoint: str | None = None,
        deployment_name: str | None = None,
        threshold: float = 3.0,
    ):
        load_dotenv()

        self.endpoint = endpoint or os.getenv("AZURE_AI_PROJECT_ENDPOINT")
        self.deployment_name = deployment_name or os.getenv("EVAL_MODEL_DEPLOYMENT", "gpt-4.1")
        self.threshold = threshold

        if not self.endpoint:
            raise ValueError(
                "Azure AI Project endpoint required. "
                "Set AZURE_AI_PROJECT_ENDPOINT env var or pass endpoint argument."
            )

        self._client = None
        self._openai_client = None

    def _get_client(self):
        """Get or create the AI Project client."""
        if self._client is None:
            try:
                from azure.ai.projects import AIProjectClient
                from azure.identity import DefaultAzureCredential
            except ImportError:
                raise ImportError(
                    "Required packages missing. Install with:\n"
                    "  pip install azure-ai-projects azure-identity"
                )
            self._client = AIProjectClient(
                endpoint=self.endpoint,
                credential=DefaultAzureCredential(),
            )
        return self._client

    def _get_openai_client(self):
        """Get OpenAI client from the AI Project client."""
        if self._openai_client is None:
            self._openai_client = self._get_client().get_openai_client()
        return self._openai_client

    # -- Eval API methods ---------------------------------------------------

    def create_eval(self, name: str, metrics: list[str]) -> str:
        """
        Create an evaluation definition with the specified metrics.

        Returns the eval ID.
        """
        client = self._get_openai_client()

        testing_criteria = []
        for metric in metrics:
            config = FOUNDRY_METRICS[metric]
            testing_criteria.append({
                "type": "azure_ai_evaluator",
                "name": metric,
                "evaluator_name": config["evaluator_name"],
                "initialization_parameters": {
                    "deployment_name": self.deployment_name,
                },
                "data_mapping": config["data_mapping"],
            })

        # Build schema from all data_mapping fields
        all_fields = set()
        for metric in metrics:
            for key in FOUNDRY_METRICS[metric]["data_mapping"]:
                all_fields.add(key)

        properties = {}
        for f in all_fields:
            properties[f] = {
                "anyOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "object"}},
                ]
            }

        eval_obj = client.evals.create(
            name=name,
            data_source_config={
                "type": "custom",
                "item_schema": {
                    "type": "object",
                    "properties": properties,
                },
                "include_sample_schema": True,
            },
            testing_criteria=testing_criteria,
        )

        return eval_obj.id

    def run_eval(
        self,
        eval_id: str,
        items: list[dict],
        run_name: str | None = None,
    ) -> str:
        """
        Submit items for evaluation. Returns the run ID.
        """
        try:
            from openai.types.evals.create_eval_jsonl_run_data_source_param import (
                CreateEvalJSONLRunDataSourceParam,
                SourceFileContent,
                SourceFileContentContent,
            )
        except ImportError:
            raise ImportError(
                "openai package with evals support required.\n"
                "  pip install openai>=1.50.0"
            )

        client = self._get_openai_client()

        content_items = []
        for item in items:
            clean = {k: v for k, v in item.items() if not k.startswith("_")}
            content_items.append(SourceFileContentContent(item=clean))

        eval_run = client.evals.runs.create(
            eval_id=eval_id,
            name=run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            data_source=CreateEvalJSONLRunDataSourceParam(
                type="jsonl",
                source=SourceFileContent(
                    type="file_content",
                    content=content_items,
                ),
            ),
        )

        return eval_run.id

    def wait_for_completion(
        self,
        eval_id: str,
        run_id: str,
        timeout: int = 300,
        poll_interval: int = 5,
    ) -> dict:
        """Poll until eval run completes. Returns status + output items."""
        client = self._get_openai_client()
        start = time.time()

        while True:
            run = client.evals.runs.retrieve(run_id=run_id, eval_id=eval_id)

            if run.status in ("completed", "failed"):
                output_items = list(client.evals.runs.output_items.list(
                    run_id=run_id,
                    eval_id=eval_id,
                ))
                return {
                    "status": run.status,
                    "output_items": output_items,
                    "report_url": getattr(run, "report_url", None),
                    "eval_id": eval_id,
                    "run_id": run_id,
                }

            if time.time() - start > timeout:
                return {
                    "status": "timeout",
                    "output_items": [],
                    "eval_id": eval_id,
                    "run_id": run_id,
                }

            time.sleep(poll_interval)

    def _parse_results(
        self,
        output_items: list,
        metrics: list[str],
        debug: bool = False,
    ) -> list[dict]:
        """
        Parse Foundry output items into a list of score dicts.

        Returns one dict per item with {metric_name: {"score": float, "pass": bool, "reason": str}}.
        """
        parsed = []

        for i, output_item in enumerate(output_items):
            # Convert to dict
            if hasattr(output_item, "model_dump"):
                data = output_item.model_dump()
            elif hasattr(output_item, "__dict__"):
                data = output_item.__dict__
            else:
                data = output_item if isinstance(output_item, dict) else {}

            if debug:
                print(f"  [DEBUG] Item {i} keys: {list(data.keys())}")

            item_scores = {}
            results_array = data.get("results", [])

            # Parse per-evaluator results
            evaluator_outputs = {}
            if isinstance(results_array, list):
                for result_item in results_array:
                    if hasattr(result_item, "model_dump"):
                        rd = result_item.model_dump()
                    elif hasattr(result_item, "__dict__"):
                        rd = result_item.__dict__
                    else:
                        rd = result_item if isinstance(result_item, dict) else {}

                    name = rd.get("name", "")
                    sample = rd.get("sample", {})
                    output = sample.get("output", [])

                    # Parse output content (JSON or S-tag format)
                    ev_output = self._parse_evaluator_output(output)
                    if ev_output:
                        evaluator_outputs[name] = ev_output

                    if debug:
                        print(f"    Evaluator '{name}': {ev_output}")

            # Also check top-level sample.output as fallback
            top_output = data.get("sample", {}).get("output", {})
            if isinstance(top_output, list):
                top_parsed = self._parse_evaluator_output(top_output)
            elif isinstance(top_output, dict):
                top_parsed = top_output
            else:
                top_parsed = {}

            for metric in metrics:
                ev = evaluator_outputs.get(metric, top_parsed)
                item_scores[metric] = self._extract_score(metric, ev)

            parsed.append(item_scores)

        return parsed

    def _parse_evaluator_output(self, output: Any) -> dict:
        """Parse evaluator output (list of content items) to a flat dict."""
        if isinstance(output, dict):
            return output
        if not isinstance(output, list):
            return {}

        result = {}
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not content or not isinstance(content, str):
                continue
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    result.update(parsed)
            except json.JSONDecodeError:
                # S-tag format: <S0>reasoning</S0><S1>summary</S1><S2>score</S2>
                score_match = re.search(r'<S2>(\d+)</S2>', content)
                reasoning_match = re.search(r'<S1>(.*?)</S1>', content, re.DOTALL)
                if score_match:
                    result["score"] = int(score_match.group(1))
                    if reasoning_match:
                        result["explanation"] = reasoning_match.group(1).strip()
        return result

    def _extract_score(self, metric: str, ev: dict) -> dict:
        """Extract normalized score from evaluator output."""
        if not ev:
            return {"score": None, "pass": False, "reason": "No result"}

        # score 1-5 format (coherence, fluency, relevance, groundedness, intent_resolution, etc.)
        if "score" in ev:
            s = float(ev["score"])
            return {
                "score": s,
                "pass": s >= self.threshold,
                "reason": ev.get("explanation", ev.get("reasoning", ev.get("reason", ""))),
            }

        # flagged format (task_adherence): flagged=False means pass
        if "flagged" in ev:
            flagged = ev["flagged"]
            return {
                "score": 1.0 if flagged else 5.0,
                "pass": not flagged,
                "reason": ev.get("reasoning", ""),
            }

        # success bool format (task_completion, tool_success)
        if "success" in ev and isinstance(ev["success"], bool):
            return {
                "score": 5.0 if ev["success"] else 1.0,
                "pass": ev["success"],
                "reason": ev.get("explanation", ""),
            }

        # tool_calls_success_level format (tool_call_accuracy)
        if "tool_calls_success_level" in ev:
            s = float(ev["tool_calls_success_level"])
            return {
                "score": s,
                "pass": s >= self.threshold,
                "reason": ev.get("chain_of_thought", ""),
            }

        # result pass/fail (tool_input_accuracy)
        if "result" in ev:
            passed = str(ev["result"]).lower() == "pass"
            return {
                "score": 5.0 if passed else 1.0,
                "pass": passed,
                "reason": ev.get("chain_of_thought", ""),
            }

        # label pass/fail (tool_output_utilization)
        if "label" in ev:
            passed = str(ev["label"]).lower() == "pass"
            return {
                "score": 5.0 if passed else 1.0,
                "pass": passed,
                "reason": ev.get("reason", ""),
            }

        return {"score": None, "pass": False, "reason": f"Unparsed output: {ev}"}

    # -- High-level convenience methods -------------------------------------

    def evaluate_items(
        self,
        items: list[dict],
        metrics: list[str] | None = None,
        eval_name: str | None = None,
        timeout: int = 300,
        debug: bool = False,
    ) -> dict:
        """
        Evaluate a list of items and return parsed results.

        Each item should have at minimum: {"query": "...", "response": "..."}.
        Add "context" for groundedness, "tool_definitions" for agent metrics.

        Returns:
            {
                "scores": [{"coherence": {"score": 4.0, ...}, ...}, ...],
                "eval_id": "...",
                "run_id": "...",
                "report_url": "...",
                "status": "completed",
            }
        """
        metrics = metrics or ["coherence", "fluency", "relevance"]

        name = eval_name or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        _out(f"Creating Foundry evaluation **{name}** with metrics: `{metrics}`")

        eval_id = self.create_eval(name, metrics)
        _out(f"Eval created: `{eval_id}`")

        run_id = self.run_eval(eval_id, items)
        _out(f"Run submitted: `{run_id}` — waiting for completion (timeout={timeout}s)...")

        result = self.wait_for_completion(eval_id, run_id, timeout=timeout)
        _out(f"Status: **{result['status']}**")

        if result.get("report_url"):
            _out(f"[Open in Foundry portal]({result['report_url']})")

        scores = []
        if result["status"] == "completed":
            scores = self._parse_results(result["output_items"], metrics, debug=debug)

        return {
            "scores": scores,
            "eval_id": eval_id,
            "run_id": run_id,
            "report_url": result.get("report_url"),
            "status": result["status"],
        }

    def evaluate_comparison(
        self,
        report: ComparisonReport,
        metrics: list[str] | None = None,
        timeout: int = 300,
        debug: bool = False,
    ) -> dict:
        """
        Evaluate a ComparisonReport via Foundry.

        Runs both source and target model responses through Foundry evaluation,
        then returns side-by-side scores.

        Returns:
            {
                "source": {"scores": [...], "eval_id": "...", ...},
                "target": {"scores": [...], "eval_id": "...", ...},
                "summary": {"metric": {"source_avg": x, "target_avg": y, "delta": z}, ...},
            }
        """
        metrics = metrics or ["coherence", "fluency", "relevance"]

        # Build items for source and target
        source_items = []
        target_items = []
        for r in report.results:
            base = {
                "query": r.test_case.prompt,
                "context": r.test_case.context or "",
            }
            if r.test_case.expected_output:
                base["ground_truth"] = r.test_case.expected_output

            source_items.append({**base, "response": r.source_response})
            target_items.append({**base, "response": r.target_response})

        _out(f"### Foundry Evaluation: {report.source_model} vs {report.target_model}\n"
             f"{len(report.results)} test cases, metrics: `{metrics}`")

        # Evaluate source
        _out(f"#### Source model: {report.source_model}")
        source_result = self.evaluate_items(
            source_items,
            metrics=metrics,
            eval_name=f"migration_{report.source_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timeout=timeout,
            debug=debug,
        )

        # Evaluate target
        _out(f"#### Target model: {report.target_model}")
        target_result = self.evaluate_items(
            target_items,
            metrics=metrics,
            eval_name=f"migration_{report.target_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timeout=timeout,
            debug=debug,
        )

        # Build summary
        summary = {}
        for metric in metrics:
            source_scores = [
                s[metric]["score"] for s in source_result["scores"]
                if s.get(metric, {}).get("score") is not None
            ]
            target_scores = [
                s[metric]["score"] for s in target_result["scores"]
                if s.get(metric, {}).get("score") is not None
            ]
            src_avg = sum(source_scores) / max(len(source_scores), 1)
            tgt_avg = sum(target_scores) / max(len(target_scores), 1)
            summary[metric] = {
                "source_avg": round(src_avg, 2),
                "target_avg": round(tgt_avg, 2),
                "delta": round(tgt_avg - src_avg, 2),
            }

        # Display summary as markdown table
        lines = [
            f"**Foundry Evaluation Summary**\n",
            f"| Metric | Source | Target | Delta |",
            f"|--------|-------:|-------:|------:|",
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
# Workflow Trajectory (for agent evaluation logging)
# ---------------------------------------------------------------------------

@dataclass
class WorkflowStep:
    """A single step in a workflow trajectory."""
    tool_name: str
    arguments: dict
    result: dict
    success: bool
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class WorkflowTrajectoryLog:
    """
    Logs the execution trajectory of an AI workflow for evaluation.

    Captures plan, execution steps, and final answer in a format
    compatible with Azure AI Foundry agent evaluation.

    Usage:
        trajectory = WorkflowTrajectoryLog("my_workflow")
        trajectory.log_plan("user query", ["tool1", "tool2"])
        trajectory.log_step("tool1", {"arg": "val"}, {"result": "data"}, True)
        trajectory.finalize(True, "Final answer")
        trajectory.save()
    """

    workflow_id: str
    output_dir: str = "evaluation_output/trajectories"
    workflow_run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    steps: list[WorkflowStep] = field(default_factory=list)
    user_request: str = ""
    planned_tools: list[str] = field(default_factory=list)
    tool_definitions: list[dict] = field(default_factory=list)
    status: str = "in_progress"
    final_answer: str = ""

    def log_plan(
        self,
        user_request: str,
        planned_tools: list[str],
        tool_definitions: list[dict] | None = None,
    ) -> None:
        """Log the initial plan for this workflow."""
        self.user_request = user_request
        self.planned_tools = planned_tools
        self.tool_definitions = tool_definitions or []

    def log_step(
        self,
        tool_name: str,
        arguments: dict,
        result: dict,
        success: bool,
    ) -> None:
        """Log an execution step."""
        self.steps.append(WorkflowStep(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            success=success,
        ))

    def finalize(self, success: bool, final_answer: str) -> None:
        """Mark the workflow as complete."""
        self.status = "completed" if success else "failed"
        self.final_answer = final_answer

    def to_foundry_format(self) -> dict:
        """Convert to Foundry-compatible format for agent evaluation."""
        # Build message-based response format for Foundry agent evaluators
        messages = []
        for s in self.steps:
            # Tool call from assistant
            messages.append({
                "role": "assistant",
                "content": [{
                    "type": "tool_call",
                    "tool_call_id": f"call_{s.tool_name}_{s.timestamp}",
                    "name": s.tool_name,
                    "arguments": s.arguments,
                }],
            })
            # Tool result
            messages.append({
                "role": "tool",
                "content": [{
                    "type": "tool_result",
                    "tool_call_id": f"call_{s.tool_name}_{s.timestamp}",
                    "tool_result": s.result,
                }],
            })
        # Final answer
        if self.final_answer:
            messages.append({
                "role": "assistant",
                "content": self.final_answer,
            })

        return {
            "query": self.user_request,
            "response": messages,
            "tool_definitions": self.tool_definitions,
            "ground_truth": self.planned_tools,
            "context": "",
        }

    def to_dict(self) -> dict:
        """Full dict representation for saving."""
        return {
            "workflow_id": self.workflow_id,
            "workflow_run_id": self.workflow_run_id,
            "user_request": self.user_request,
            "planned_tools": self.planned_tools,
            "tool_definitions": self.tool_definitions,
            "steps": [
                {
                    "tool_name": s.tool_name,
                    "arguments": s.arguments,
                    "result": s.result,
                    "success": s.success,
                    "timestamp": s.timestamp,
                }
                for s in self.steps
            ],
            "status": self.status,
            "final_answer": self.final_answer,
            "foundry_format": self.to_foundry_format(),
        }

    def save(self, filename: str | None = None) -> str:
        """Save trajectory to JSON file."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        fname = filename or f"{self.workflow_id}_{self.workflow_run_id[:8]}.json"
        path = os.path.join(self.output_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return path


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def check_foundry_available() -> bool:
    """Check if Foundry Evals API dependencies are available."""
    try:
        from azure.ai.projects import AIProjectClient  # noqa: F401
        from azure.identity import DefaultAzureCredential  # noqa: F401
        return True
    except ImportError:
        return False


FOUNDRY_AVAILABLE = check_foundry_available()
