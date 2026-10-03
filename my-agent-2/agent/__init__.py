"""The agent package — orchestration only, no estimation or I/O of its own.

Phase 2 houses a single entrypoint, ``main.run``, which wires the harness
modules together: card → panel → statistics → simulation → the only writer.
Later phases add ``loop.py`` (decisions) and ``prompts.py`` (LLM phrasing);
all numeric work stays in ``harness/tools``.
"""
