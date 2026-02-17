"""
Prompty loader — loads .prompty files (Microsoft standard format).

Prompty is an asset class for LLM prompts that uses YAML frontmatter
for metadata and a Jinja template for the prompt body.

Usage:
    from src.evaluate.prompts import load_prompty

    prompty = load_prompty("rag")
    print(prompty["system_prompt"])  # The system prompt text
    print(prompty["metadata"])       # YAML frontmatter as dict
"""

from pathlib import Path

import yaml


_PROMPTS_DIR = Path(__file__).parent


def load_prompty(name: str) -> dict:
    """Load a .prompty file by name.

    Args:
        name: Prompty file name without extension (e.g., "rag", "classify_sentiment").

    Returns:
        {
            "system_prompt": str,   # The system prompt template
            "metadata": dict,       # Parsed YAML frontmatter (name, description, model, inputs, ...)
        }
    """
    path = _PROMPTS_DIR / f"{name}.prompty"
    if not path.exists():
        raise FileNotFoundError(f"Prompty file not found: {path}")

    raw = path.read_text(encoding="utf-8")

    # Split on YAML frontmatter delimiters (--- ... ---)
    parts = raw.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"Invalid .prompty format in {path}: missing YAML frontmatter (--- ... ---)")

    # Parse YAML frontmatter
    metadata = yaml.safe_load(parts[1]) or {}

    # Extract system prompt from template body
    body = parts[2].strip()

    # Prompty format uses "system:" prefix for system messages
    system_prompt = body
    if body.startswith("system:"):
        system_prompt = body[len("system:"):].strip()

    return {
        "system_prompt": system_prompt,
        "metadata": metadata,
    }


def list_prompty() -> list[str]:
    """List all available .prompty files."""
    return sorted(p.stem for p in _PROMPTS_DIR.glob("*.prompty"))
