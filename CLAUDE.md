# AI Shorts Factory — Build Prompt

> Hand this to a coding agent (Claude Code / Codex) as the operational spec. It can also live at the repo root as `CLAUDE.md`. Fill the `<<...>>` slots before Phase 0, or stub them and flag the gate. Do not proceed past a gate that isn't green.

---

## 0. Your role and operating rules

You are building a **supervised** AI content engine that turns a topic into a finished, source-backed, vertical YouTube Short — with a human approving before anything publishes. You are not building an autonomous publisher and not building a learning model. Not yet.

Follow these rules on every task. They override convenience.

1. **Read before you write.** Inspect the live file/state before editing it. Read the neighbouring module before adding one.
2. **Check the world; don't assume it.** Versions, paths, API flags, quotas, what's actually in the file/DB. Verify in the same turn you assert it. YouTube quota/policy and API shapes change — confirm against live docs, don't trust this file's summary of them.
3. **A failure is a hypothesis.** Confirm it's real and name the cause before changing anything. First plausible explanation is usually wrong.
4. **Prove, don't eyeball.** For media, "it rendered" is not proof. `ffprobe` the output and assert resolution/duration/streams. Green tests are the floor.
5. **Verify edits in a separate pass.** That pass is where your mistakes live.
6. **Make verification adversarial.** Test the refusal, not just the success (e.g. a red-classified topic must be *blocked*, prove it is). Convergence of independent checks is the signal; one confirmatory glance is worth nothing.
7. **The plan is the spine.** Externalize multi-step work as tracked tasks before the first edit. Every edit traces to a task. No opportunistic detours.
8. **Hold the scope line.** Build the phase you're on. Flag adjacent work; never silently expand. Constraints below are the task, not obstacles.
9. **Match the house style.** Mirror existing patterns before inventing. New code reads as if the existing author wrote it.
10. **Report honestly.** Separate verified from assumed, name what didn't finish, give rollback commands. Under-claiming beats false closure.
11. **No generic frontends.** When you reach the dashboard, use the appropriate design skill and reference real patterns. Do not ship a templated default. Show the design intent before building it and get sign-off.

---

## 1. Inputs to confirm before writing code

These are **gates**, not preferences. Stub each with a clearly-marked placeholder if unknown, and list every stub in your first report.

| Input | Value | Notes |
|---|---|---|
| Book file (legally owned) | `<<path/to/book.pdf or .txt>>` | Research/inspiration only. See §5 copyright. |
| Output language | `<<e.g. English>>` | Narration + captions. |
| Visual style | `<<e.g. illustrated realism, diagram-forward>>` | Must be reproducible across videos, not per-video reinvention. |
| LLM provider/model | `<<>>` | Script + idea + fact-extraction. |
| TTS provider/voice | `<<>>` | Licensed or owned/cloned voice only. |
| Image gen provider | `<<>>` | Commercial-use license required. |
| Music/SFX source | `<<>>` | Commercial rights required. |
| YouTube channel + OAuth | `<<>>` | For upload + Analytics API. |
| Budget ceiling per video | `<<$>>` | Enforce a hard cost cap per render in code. |

---

## 2. Build this first — the walking skeleton (Phase 0)

**One hardcoded path, no abstractions, no dashboard, no database.** Topic is hardcoded to `soap`. One command runs it end to end and produces one MP4.

Build it in this inner order so you prove the cheap, deterministic parts before spending on APIs:

1. **Hardcode a research brief.** A single `soap.brief.json` you write by hand: 4–6 paraphrased, cited claims. No retrieval yet. (Retrieval is Phase 1.)
2. **Script from brief.** One LLM call → `soap.script.json`: a 40–50s script broken into scenes, each with `{ narration, caption, duration, visual_prompt, source_claim_id }`. Schema-validate the output.
3. **Assemble with placeholder frames FIRST.** Before any image API: render each scene as a solid-color 1080×1920 frame with the caption burned in, sliced to the TTS audio. Prove FFmpeg/Remotion assembly is deterministic and correct with zero image spend.
4. **TTS.** Narration → per-scene audio → concatenated track. Loudness-normalize (target ~ -14 LUFS).
5. **Swap placeholders for generated images.** One image per scene from the image API, same pipeline.
6. **Emit `soap.mp4`.**

**Skeleton acceptance criteria (all must pass, proven by command output):**

- [ ] `ffprobe soap.mp4` shows `1080x1920`, duration within the scripted total ±0.5s, one audio + one video stream.
- [ ] Captions are present and legible inside safe margins (assert via the caption timing file + one rasterized frame check).
- [ ] Audio integrated loudness within ±1 LU of target (`ffmpeg ... loudnorm print_format=json`).
- [ ] Total API spend for one run is logged and under the per-video cap.
- [ ] Re-running with the same inputs produces a byte-stable render *or* a documented reason it can't (image gen non-determinism is acceptable; assembly must be stable).

**Gate:** Do not start Phase 1 until every box is checked in a report.

---

## 3. Target architecture (extract only after the skeleton proves the path)

Six systems. Extract abstractions from the soap path plus one second real topic (`roman concrete`) — do not design interfaces from imagination.

1. **Knowledge & research** — ingest book → chunk → tag → retrieve per topic → extract discrete claims → verify each against ≥1 independent source → store citation + confidence. Blocks substantial book-text reproduction.
2. **Idea generator** — N ranked concepts per topic with 5 hook variants each, payoff, series, visual-potential score, safety class, source availability, similarity-to-recent.
3. **Script & storyboard** — brief → structured 40–50s script → per-scene storyboard (narration / caption / visual prompt / camera / SFX / duration / source id).
4. **Media production** — TTS, image/video gen, captions, licensed audio, deterministic FFmpeg/Remotion assembly, 1080×1920, loudness norm, cover frame.
5. **Review & publish dashboard** — pipeline stages, per-video edit/regenerate-one-scene/inspect-sources, preview, approve/reject, private upload, schedule. Human-approved publish only.
6. **Analytics & learning** — YouTube Analytics (retention, `audienceWatchRatio`, `subscribersGained`, etc.) + experiment ledger + rule-based scoring. No ML until enough videos exist.

**Supporting infra:** PostgreSQL, object storage for sources/media, a background job queue for gen+render, cost/usage tracking, retryable job logs, secrets manager, backups.

---

## 4. Phased plan with gates

Each phase ships behind an acceptance test. Do not cross a gate on assumption.

| Phase | Deliverable | Gate (must be provable) |
|---|---|---|
| 0 | Walking skeleton (§2) | §2 acceptance criteria all green |
| 1 | Book ingestion + retrieval + claim verification + citation store | Retrieval returns cited passages for soap & concrete; a fabricated claim is rejected by the verifier (adversarial check) |
| 2 | Idea → hook → script generation (replaces hand-written brief) | Generated script for a new topic passes schema + every claim carries a source id |
| 3 | Storyboard + real visual-prompt generation | A third topic renders end-to-end with generated visuals under cost cap |
| 4 | Review dashboard (§0 rule 11) | Human can edit a hook, regenerate one scene, and preview without a full re-render |
| 5 | Private YouTube upload + synthetic-content disclosure flag set | Video lands as private/unlisted; disclosure flag confirmed via API response, not assumed |
| 6 | Analytics ingestion + experiment ledger + rule-based scoring | Real retention/subscriber data lands per video; scores reproduce by hand for one video |
| 7 | Controlled scheduling / graduated autonomy | Only after ~30–50 human-reviewed videos; red/health/safety topics stay human-gated permanently |

---

## 5. Hard constraints (non-negotiable)

**Copyright.** The book is research and inspiration only. Owning a copy does not grant adaptation rights. Paraphrase; never reproduce substantial text or any artwork. Every on-screen or narrated claim must trace to a stored citation. Independently verify important claims against a second source.

**Safety classification.** Every idea gets a class before it can render:
- **Green** (concrete, rope, pottery, water filtration, crop rotation): proceed.
- **Yellow** (soap, furnaces, electricity, food preservation): requires source verification + on-screen caution; never rely on the book alone for medical/survival specifics.
- **Red** (weapons, explosives, gasoline/fuel synthesis, toxic chemistry, unsafe medicine): **no actionable instructions.** Treat only as non-actionable historical/scientific context, or exclude. Build the classifier so a red topic is *blocked from the procedural pipeline*, and prove the block (rule 6).

**YouTube disclosure.** Realistic synthetic scenes must set YouTube's altered/synthetic-content flag on upload. Verify the flag from the API response.

**Human-in-the-loop.** Publishing stays human-approved until quality is proven across the first ~30–50 videos. Health, safety, dangerous-technical, real-location-disaster, copyright-sensitive, and realistic-depiction content stay human-gated indefinitely.

**Secrets & cost.** No keys in code or logs. Enforce the per-video budget cap in code; abort a render that would exceed it.

---

## 6. First ten pilot videos (visual-strong, non-dangerous)

Soap · Roman concrete · Apple cider vinegar · Charcoal · Pottery · Rope · Water filtration · Basic compass · Food preservation · Simple mechanical water pump.

Vary **one element at a time** (usually the hook) across a series so the experiment ledger produces comparable data instead of noise.

---

## 7. Reporting format I expect from you

End every work session with:

- **Done (verified):** what you proved, with the command/output that proves it.
- **Done (assumed):** anything you couldn't verify and why.
- **Not finished:** open items, with the current gate status.
- **Diffs:** surgical before/after for each change, each tied to a task.
- **Rollback:** exact commands to undo this session.

---

## 8. Explicitly out of scope for v1

Autonomous publishing · ML ranking model · multi-language · competitor scraping at scale · desktop app · anything not required to get ten accurate, engaging, human-approved videos out the door.

**First milestone, restated:** one complete, original, source-backed *"How to reinvent soap if civilization collapsed"* Short, from the book file to a verified 1080×1920 MP4, under the cost cap.
