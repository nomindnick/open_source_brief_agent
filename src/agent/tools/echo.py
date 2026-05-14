"""``EchoTool`` — a stub tool that returns its input verbatim.

Exists so Sprint 2.2's agent loop can be exercised end-to-end without
any HTTP calls, subprocess wrapping, or filesystem touchpoints. Useful
later for prompt iteration too: swap the real tool registry for one
containing only EchoTool to test loop / parser changes in isolation.
"""

from __future__ import annotations

from typing import Any

from agent.tools.base import Tool


class EchoTool(Tool):
    name = "echo"
    description = "Returns the input message verbatim. Useful for testing the agent loop."
    input_schema = {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "The text to echo back."}
        },
        "required": ["message"],
    }

    def run(self, input: dict[str, Any]) -> str:
        msg = input.get("message")
        if not isinstance(msg, str):
            return "ERROR: echo expects an object with a string 'message' key."
        return msg
