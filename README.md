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

## OpenAI Real Model Call (gpt-5.4)

Set environment variables before running:

```bash
export OPENAI_API_KEY="<your_key>"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-5.4"
export OPENAI_REASONING_EFFORT="xhigh"
export OPENAI_STORE="false"
```

For OpenAI-compatible gateways, set:

```bash
export OPENAI_BASE_URL="https://your-gateway"
export OPENAI_RESPONSES_URL="https://your-gateway/responses"
```

## Telegram Bridge

Run Hermes from Telegram chat via long polling:

```bash
export TELEGRAM_BOT_TOKEN="<bot_token>"
export TELEGRAM_ALLOWED_CHAT_IDS="123456789"
export HERMES_MAX_STEPS="8"
hermes-agent-telegram
```

Optional:

```bash
export HERMES_ALLOW_GUARDED="1"
hermes-agent-telegram --allow-guarded-from-env
```

## Run Tests

```bash
pytest -q
```
