# Stack report

This was a brand-new, empty repository before this session — no pre-existing
code or conventions to match. Choices below, made explicit per CLAUDE.md rule 9.

## Runtime

- **Python 3.13.3** (`/opt/homebrew/bin/python3`), project-local venv at `.venv/`.
- **ffmpeg / ffprobe 8.1.2** (`/opt/homebrew/bin/`), invoked via `subprocess`
  — no Python ffmpeg wrapper library; direct CLI calls, argument lists kept
  explicit so every ffmpeg invocation is copy-pasteable for debugging.
- **jsonschema 4.26.0** — formal JSON Schema validation for brief/script.
- **Pillow 12.3.0** — frame/caption rendering (safe-margin caption cards,
  gradient placeholder images).
- **python-dotenv 1.2.2** — `.env` loading for CLAUDE.md §1 inputs.
- **pytest 9.1.1** — test runner.

Pinned in `requirements.txt` (from `pip freeze`).

## Layout

```
ai-shorts-factory/
  CLAUDE.md                 # the build spec this implements
  .env.example               # documents every CLAUDE.md §1 input
  run.sh                     # the one command: ./run.sh soap
  requirements.txt
  data/soap/soap.brief.json  # hand-authored Phase 0 brief
  schemas/                   # brief.schema.json, script.schema.json
  src/shorts_factory/
    config.py                # resolves .env inputs, flags stubs
    safety.py                # green/yellow/red classifier, fail-closed
    schema_validate.py       # jsonschema + cross-document checks
    cost_tracker.py          # CostTracker + BudgetGuard (pre-request abort)
    captions.py               # safe-margin caption rendering (Pillow)
    assembly.py               # deterministic ffmpeg pipeline
    verify.py                 # ffprobe/loudnorm-based acceptance checks
    pipeline.py                # orchestrator + CLI entrypoint
    providers/
      llm.py, tts.py, image.py # ABC + StubXProvider per provider kind
  tests/                       # pytest, see below
  artifacts/<topic>/           # gitignored render output (per-run)
```

## Architecture decision: provider interfaces, stub-first

CLAUDE.md §1 lists LLM/TTS/image providers as required, but the operating
rules (§0) forbid paid calls before approval, and the user's own guidance
said to build the whole pipeline with no paid APIs first. Rather than
special-casing "no providers yet," every provider kind (`LLMProvider`,
`TTSProvider`, `ImageProvider`) is an ABC with one concrete implementation
today: `StubXProvider` — deterministic, local, zero network, zero cost, but
routed through the exact same `CostTracker.check_budget()` / `.record()`
calls a real provider will use. This means:

- The full pipeline runs today, for free, and is fully tested.
- Paid generation routes through one shared fal.ai gateway selected with
  `LLM_PROVIDER=fal`, `TTS_PROVIDER=fal`, and `IMAGE_PROVIDER=fal`.
- The budget-guard code path is exercised now (tests/test_budget.py), not
  left untested until a real paid provider exists.

## Test suite (21 tests, all passing)

```
PYTHONPATH=src python3 -m pytest tests/ -v
```

- `test_schema.py` — valid brief/script pass; missing field, wrong type,
  over-length caption, missing citation, out-of-window duration all rejected.
- `test_safety.py` — soap=yellow, roman concrete=green, gunpowder=red
  (including a rephrased "how to make gunpowder at home"), unknown topic
  fails closed to red, yellow topics are not blocked.
- `test_budget.py` — a call within budget succeeds; a call that would push
  total spend over the cap raises `BudgetExceeded` **and the simulated
  provider's call counter stays at zero** — proves the abort happens before
  the request, not after.
- `test_determinism.py` — two independent assembly runs on identical local
  inputs produce byte-identical output (sha256 match).
- `test_pipeline_integration.py` — `run_pipeline("gunpowder")` is blocked
  with zero artifacts created; `run_pipeline("soap")` runs end-to-end and
  every verification criterion passes.
