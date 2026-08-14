# AI Shorts Factory

A supervised pipeline for producing source-backed vertical Shorts. Rendering,
review, private YouTube upload, analytics, and Telegram review are separate
steps. Nothing publishes autonomously.

## Safe setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Keep `.env`, the YouTube OAuth files, and `data/private/` out of Git. The book
index is private research material and its chunks are never copied into a
script or shown in the dashboard.

## Providers

The free deterministic mode uses `stub` providers. A publishable run currently
supports:

- `LLM_PROVIDER=anthropic`, with `ANTHROPIC_API_KEY` and `LLM_MODEL`
- `TTS_PROVIDER=elevenlabs`, with `ELEVENLABS_API_KEY` and `TTS_VOICE`
- `IMAGE_PROVIDER=fal`, with `FAL_KEY` and `IMAGE_MODEL`
- `SEARCH_PROVIDER=tavily`, with `SEARCH_API_KEY`

Before enabling any paid provider, set `BUDGET_CAP_USD` and conservative values
for `LLM_COST_PER_SCRIPT_USD`, `TTS_COST_PER_1K_CHARS_USD`, and
`IMAGE_COST_PER_IMAGE_USD`. Every request checks the projected total before it
is sent. These configured estimates are a safety ceiling ledger, not a provider
invoice; reconcile them with the provider billing dashboard.

## Workflow

```bash
./retrieve.sh soap      # private book index + independent web citations
./run.sh soap           # script, TTS, images, assembly, verification
./dashboard.sh          # local review dashboard at 127.0.0.1:8420
./telegram.sh           # optional supervised Telegram review bot
PYTHONPATH=src .venv/bin/python -m shorts_factory.publish soap
./analytics.sh          # fetch summary metrics and retention curves
```

Publishing requires `approved` review state, uploads privately, sets and then
reads back YouTube's synthetic-media disclosure, and permits only one successful
upload per UTC day. A failed upload is recorded and can be retried. Uploads are
added to the experiment ledger automatically.

Telegram requires `TELEGRAM_BOT_TOKEN` and a comma-separated numeric
`TELEGRAM_ALLOWED_USER_IDS` allowlist. It supports `/status`, `/video`,
`/approve`, `/reject`, and explicit `/publish`. It has no generation command.

## Pilot topics

Soap, Roman concrete, apple cider vinegar, charcoal, pottery, rope, water
filtration, basic compass, food preservation, and a simple mechanical water
pump are configured. Unknown or dangerous topics fail closed before providers
are called. Yellow topics receive an on-screen caution.

## Verification

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests -q
ffprobe -v error -show_streams -show_format artifacts/soap/soap.mp4
```

Local tests do not prove live provider credentials, source quality, YouTube
OAuth, or channel analytics. Those gates require explicit credentials and can
incur costs, so they are deliberately not exercised by the test suite.
