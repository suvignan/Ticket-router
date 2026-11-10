# API contract

Behaviour decided **before** the code (same rule as the model gate). Numbers come from the training data
(`data/splits/train.parquet`, 367,414 narratives) unless stated.

## `POST /predict`

**Request:** `{"text": "..."}`. The model is **one per deployment** (`params.yaml → serving.model`, `lr` or `bert`), not
per request: loading both would double the memory, and a switch is a deliberate redeploy, visible in `/version`.

Checks run in this order: **413** → **no content** → **too short** → model.

**Why 413 is first (privacy, decided 2026-11-10):** an oversized request is rejected having touched **only its
length**: it isn't hashed, cleaned or logged beyond `text_len`, and its row has `text_sha256 = NULL`. The hash exists
to trace a *routed* ticket (duplicates, disputes); a rejected text was never routed, so hashing it has no purpose,
and data minimisation says don't compute what you don't need. A hash of personal text is still pseudonymous
personal data, not anonymous. (The server did have to receive the body to measure it; rejecting on the
`Content-Length` header before reading it is the next step if abuse shows up.)

| Limit | Value | Reason |
|---|---|---|
| Max length | **35,000 characters** (raw) → **413** | **Protects the service** (memory, CPU, abuse on a public endpoint). Set above the longest real complaint (32,962 chars; p99.9 = 18,492), so no genuine ticket is rejected. (Not about truncation: BERT already reads only the first 350 words by design, see `truncated_for_model`.) |
| Too short | **< 10 characters after `clean_text`** → `review`, reason `too_short`, model **not called** | The shortest training narrative is 9 chars ("paid debt"); only 1 of 367,414 is under 10. Below that it's shorter than anything the model learned from. Vague short texts that pass this rule are the confidence rule's job, not this one's. |
| No content | Ingestion's junk rules (`params.yaml → junk_rules`: redaction-only, short "see attached") → `review`, reason `no_content`, model **not called** | The same rules that kept these rows out of training. The model has never seen them, so any prediction would be a guess. One definition, reused. |

**Response (200):**

| Field | Type | Meaning |
|---|---|---|
| `decision` | `"auto_route"` \| `"review"` | Route automatically, or send to a human |
| `team` | string \| `null` | Predicted label. `null` only for `too_short` / `no_content` |
| `confidence` | float \| `null` | Top probability |
| `top2` | `[{team, probability}, ...]` \| `null` | Two most likely teams: a reviewer usually picks one of these |
| `reason` | `null` \| `"too_short"` \| `"no_content"` \| `"low_confidence"` | Why it went to review |
| `model` | `"lr"` \| `"bert"` | Which model answered |
| `model_sha256` | string (12 chars) | Audit trail in **every** response: which exact file decided |
| `threshold` | float | The confidence threshold that was applied |
| `truncated_for_model` | bool | `true` when BERT read only the first 350 words of a longer text (always `false` for LR, which reads all of it) |
| `request_id` | string (UUID4) | Traces a disputed routing to its log row |

### Review rules (applied in this order)

1. `too_short` or `no_content` → `review`, without calling the model.
2. `confidence < threshold` → `review`, reason `low_confidence`. `team` + `top2` are still returned as **suggestions**.
3. Otherwise → `auto_route`.

**Thresholds are per model:**

| Model | Threshold | Source |
|---|---|---|
| LR (`logreg_b`) | **0.6** | Chosen on validation, Day 8 |
| BERT | **0.83** | Chosen on validation, Day 19 (rule below). Its probabilities are calibrated differently: reusing 0.6 would review only 5.1% |

Both live in `params.yaml → serving.thresholds`.

**BERT threshold selection rule (written 2026-11-06, BEFORE computing BERT's validation probabilities):**
choose the threshold at which BERT sends the **same share of validation tickets to review as LR does at 0.6**
(LR's share recomputed on the same validation set, ~14.6%). Same staff workload, so the comparison is fair:
the better model is the one that **catches more of its own errors** in that same review queue.
Validation only, never test. Rounded to 2 decimals.

**Result** (`scripts/bert_threshold.py`, 142,407 validation tickets; BERT val macro-F1 0.7206 = Day 12's 0.7205):

| Model | Threshold | Sent to review | Errors caught | Caught per 1% reviewed | Accuracy of auto-routed |
|---|---|---|---|---|---|
| LR | 0.60 | 14.6% | 44.5% | 3.04 | 94.5% |
| BERT | 0.50 | 1.2% | 9.8% | 8.19 | 93.0% |
| BERT | 0.60 | 5.1% | 23.8% | 4.69 | 93.8% |
| BERT | 0.70 | 8.7% | 37.7% | 4.32 | 94.8% |
| BERT | 0.80 | 13.0% | 51.3% | 3.96 | 95.7% |
| **BERT** | **0.83** | **14.5%** | **55.4%** | **3.83** | **96.0%** |
| BERT | 0.90 | 21.4% | 65.6% | 3.07 | 96.6% |

At the same review workload, BERT catches 55.4% of its errors versus LR's 44.5%, and auto-routes at 96.0% accuracy versus 94.5%.

## `GET /health`

`200 {"status": "ok"}` when the model files are loaded, `503` otherwise. **No model call**: Render polls it often,
so it must cost nothing.

## `GET /version`

What exactly is running, in one call: model names, **sha256 of every loaded file** (from `models/manifest.json`,
verified at start-up: the service **refuses to start** on a mismatch, like the gate), the git commit the image was
built from, and the active thresholds.

## Errors

| Code | When | Body |
|---|---|---|
| 422 | `text` missing or not a string (Pydantic, automatic) | FastAPI's validation detail |
| 413 | `text` over 35,000 characters | `{"detail": "...", "request_id": "..."}` |
| 500 | Anything unexpected | `{"detail": "internal error", "request_id": "..."}`. **Never** a stack trace: that's logged server-side under the same `request_id` |

## Privacy (data minimisation, Day 1)

The raw complaint is personal data: **it's never logged or stored.** Each request logs only: `request_id`,
`sha256(text)` (duplicates and traces still work; not for 413, see above), `len(text)`, model, decision, reason,
confidence, latency. The response echoes nothing from the text.

## Prediction store (Neon Postgres)

Table `predictions` (`src/prediction_log.py`): one row per `/predict` response (200 and 413) with the same fields as the
log line, plus `team` and `model_sha256`. There is **no text column, by design.** `DATABASE_URL` comes from the
environment, never from the repo; unset means rows are not stored (stdout log only).

**Availability decision:** the insert runs **after the response is sent** (a background task, adding no latency), and if
it fails, the error is logged (`db_error` + `request_id`) and **the prediction is still returned**. A down database must
never stop tickets being routed: losing an audit row is recoverable from the stdout log, while a routing outage
isn't. Tested in `tests/test_api.py::test_database_down_still_routes`.
