"""Loading a plan: the list of goals a run works through.

A plan is JSON so a planner -- an LLM, a script, a person editing a file -- can produce one
without touching Python. Each entry is one goal in whatever vocabulary its controller takes,
and both controllers share the `stop` vocabulary from `mcagents.agents.base`:

    [
      {"point": "tree", "interaction": "Mine", "stop": {"item": "log", "count": 3}},
      {"instruction": "Craft a crafting table.", "stop": {"item": "crafting_table"}}
    ]
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

Plan = List[Dict[str, Any]]


def load_plan(path: Optional[str], default: Plan) -> Plan:
    """The plan at `path`, or `default` if no path was given."""
    if not path:
        return default
    entries = json.loads(Path(path).read_text())
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        raise SystemExit(f"{path} must hold a JSON list of goal objects")
    return entries
