# ShopSphere — AI Assistance Log

Index of the AI-assisted build history for ShopSphere. The transcripts are
stored as per-session, timestamped JSONL files in this folder and updated with every prompt.

## Format

Each `.jsonl` file is named from the session's UTC start time
(`YYYY-MM-DDTHH-MM-SS.ffffZ.jsonl`) and follows the OpenGlimpse
`ai/ryan/ai-logs` format: one raw part object per line, in chronological order.
A session contains the user `text` parts plus the assistant parts that respond
to them — `reasoning`, `tool` calls and results, `text` replies, and the
`step-start` / `step-finish` markers.

```json
{"type": "text", "text": "the full prompt or response text"}
{"type": "reasoning", "text": "assistant reasoning"}
{"type": "tool", "tool": "read", "callID": "…", "state": {"status": "completed", "input": {}, "output": "…"}}
{"type": "step-start"}
{"type": "step-finish", "reason": "tool-calls", "tokens": {}, "cost": 0}
```
