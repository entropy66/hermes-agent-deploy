# Hermes Self-Evolving Agent MVP

A runnable MVP for a JARVIS-style, self-evolving agent loop:

`Observe -> Plan -> Act -> Reflect -> Skillize`

## What This Includes

- `AgentLoop.run(task, context) -> TaskResult`
- Three-tier memory store:
  - short-term session memory
  - long-term task memory
  - skill memory
- Skill factory and versioned skill registry:
  - `propose / evaluate / publish`
- Model routing and fallback:
  - local-preferred for simple/low-risk steps
  - cloud-preferred for complex/high-risk steps
- High-autonomy executor with guardrails:
  - risk classification
  - dangerous command blocking
  - guarded action policy
  - rate limiting
  - kill switch
  - context snapshot rollback

## Install

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e .[dev]
```

## Run CLI

```bash
hermes-agent "summarize build failures and propose next step" --rounds 3
```

Allow guarded actions (still blocks hard-dangerous commands):

```bash
hermes-agent "clean temp files carefully" --allow-guarded
```

## Run Tests

```bash
pytest -q
```
