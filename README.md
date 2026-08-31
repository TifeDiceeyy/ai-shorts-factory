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

On Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
```

`ffmpeg`/`ffprobe` are provided via the `imageio_ffmpeg` package (pinned in
`requirements.txt`) and put on `PATH` automatically at import time — no
separate ffmpeg install is required on either OS.

Keep `.env`, the YouTube OAuth files, and `data/private/` out of Git. The book
index is private research material and its chunks are never copied into a
script or shown in the dashboard.

## Providers

The free deterministic mode uses `stub` providers. A publishable run currently
supports one generation gateway:

- `LLM_PROVIDER=fal`, with `FAL_LLM_ENDPOINT` and `LLM_MODEL`
- `TTS_PROVIDER=fal`, with `TTS_MODEL` and `TTS_VOICE`
- `IMAGE_PROVIDER=fal`, with `IMAGE_MODEL`
- One `FAL_KEY` authenticates all three generation adapters
- `SEARCH_PROVIDER=tavily`, with `SEARCH_API_KEY`

Before enabling any paid provider, set `BUDGET_CAP_USD` and conservative values
for `LLM_COST_PER_SCRIPT_USD`, `TTS_COST_PER_1K_CHARS_USD`, and
`IMAGE_COST_PER_IMAGE_USD`. Every request checks the projected total before it
is sent. These configured estimates are a safety ceiling ledger, not a provider
invoice; reconcile them with the provider billing dashboard.

## Video generation (defaults from Aug 2026)

Every final render now uses **layered sticker animation** and **typewriter
lyrics** automatically — no extra flags per run.

| What you get | Default | Override |
|---|---|---|
| On-screen lyrics | Character-by-character typewriter, synced to narration | `CAPTION_ANIMATION_MODE=punch` (legacy bounce) |
| Sticker entrances | Smooth fade-in (no bounce) | `ENTRANCE_STYLE=pop` (legacy overshoot) |
| Sticker count | **12–15** individually generated PNGs per video | `STICKER_TARGET_MIN` / `STICKER_TARGET_MAX` |
| Scene motion | Per-sticker idle loops (`float`, `flicker`, `drift`, `spin`, `breathe`) | Set per sticker in `script.json` → `scenes[].stickers[]` |
| Animation mode | `sticker` (still-image compositor, zero video-gen cost) | `ANIMATION_MODE=ai_video` (legacy Kling I2V, costs extra) |

### `.env` checklist before a real (paid) run

```env
BUDGET_CAP_USD=2.00          # required before any non-stub provider runs
FAL_KEY=your_key             # one key for LLM + TTS + image
LLM_PROVIDER=fal
TTS_PROVIDER=fal
IMAGE_PROVIDER=fal
IMAGE_COST_PER_IMAGE_USD=0.04  # budget guard uses this × ~15 stickers + hero
CAPTION_ANIMATION_MODE=typewriter
ENTRANCE_STYLE=fade
TYPEWRITER_CURSOR=true
STICKER_TARGET_MIN=12
STICKER_TARGET_MAX=15
ANIMATION_MODE=sticker
```

Budget note: a typical video now generates **1 hero image + 12–15 sticker
images** (not just one image per scene). Confirm `BUDGET_CAP_USD` covers
`IMAGE_COST_PER_IMAGE_USD × (sticker count + 1)` before running.

Stub mode (`LLM/TTS/IMAGE_PROVIDER=stub`) still works with zero API spend — useful
for testing assembly and timing locally.

## Workflow

```bash
./retrieve.sh soap      # private book index + independent web citations
./run.sh soap           # script, TTS, images, assembly, verification
./dashboard.sh          # local review dashboard at 127.0.0.1:8420
./telegram.sh           # optional supervised Telegram review bot
PYTHONPATH=src .venv/bin/python -m shorts_factory.publish soap
./analytics.sh          # fetch summary metrics and retention curves
```

Windows equivalents (`.ps1` wrappers, same arguments):

```powershell
.\retrieve.ps1 soap
.\run.ps1 soap               # generates script + 12-15 stickers + typewriter MP4
.\dashboard.ps1
.\telegram.ps1
$env:PYTHONPATH="src"; .venv\Scripts\python -m shorts_factory.publish soap
.\analytics.ps1
```

### Step-by-step (Windows, first video)

```powershell
cd C:\ai-shorts-factory
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
# Edit .env: set BUDGET_CAP_USD, FAL_KEY, and providers (see checklist above)

.\retrieve.ps1 soap          # optional if citations already exist in data/soap/
.\run.ps1 soap                # output: artifacts\soap\soap.mp4
.\dashboard.ps1               # review at http://127.0.0.1:8420 → approve
$env:PYTHONPATH="src"; .venv\Scripts\python -m shorts_factory.publish soap
```

Output artifacts for each topic live under `artifacts/<topic>/`:

- `<topic>.mp4` — final video (1080×1920, 40–50 s)
- `<topic>.script.json` — includes `scenes[].stickers[]` manifest
- `generated/stickers/stk-*.png` — individual sticker assets
- `verification-report.json` — must pass before publish
- `captions.srt` — subtitle file synced to narration cues

Publishing requires `approved` review state, uploads privately, sets and then
reads back YouTube's synthetic-media disclosure, and permits only one successful
upload per UTC day. A failed upload is recorded and can be retried. Uploads are
added to the experiment ledger automatically.

Telegram runs on aiogram 3 and requires `TELEGRAM_BOT_TOKEN` and a comma-separated numeric
`TELEGRAM_ALLOWED_USER_IDS` allowlist. It supports `/status`, `/video`,
`/approve`, `/reject`, and explicit `/publish` for review, plus `/plan` — a
guided flow that picks or proposes a topic, chooses a mascot, generates ideas,
and runs the full generation pipeline from inside the chat.

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
