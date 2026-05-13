"""XML tool-call parser.

Extracts ``<tool_use>`` and ``<final_answer>`` blocks from model output.
Lenient about whitespace, trailing commas, and quote style — parse errors
become observations the model can recover from.

Filled in by Sprint 2.1.
"""
