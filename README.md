![CI](https://github.com/suvignan/Ticket-router/actions/workflows/ci.yml/badge.svg)

# Ticket Router

Routes a customer complaint to the right product team by reading its text, and sends uncertain ones to a human.
It's trained on the public **CFPB Consumer Complaint** dataset (2.9 M rows), which stands in for a bank's support tickets.

- **Input:** `Consumer complaint narrative`. **Output:** `Product` (11 teams after label cleanup)
- **Metric:** macro-F1. The biggest class is 67% of rows, so accuracy alone is misleading.
- **Live:** https://ticket-router-hjiq.onrender.com (`/predict`, `/health`, `/version`). The API contract is in [docs/api.md](docs/api.md).

## Results

Time-based split: train 2023–2024, val Jan–Feb 2025, test Mar–May 2025 (opened once, Day 13).

| Model | Val macro-F1 | Test macro-F1 | Test acc | Test macro-F1, deduplicated |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.696 | 0.656 | 0.910 | 0.659 |
| LR, first 350 words | 0.694 | 0.656 | 0.909 | 0.659 |
| LSTM | 0.674 | 0.644 | 0.905 | 0.649 |
| **BERT** (`bert_small_en_uncased`, ONNX) | **0.721** | **0.686** | **0.916** | **0.686** |

BERT beats LR by +0.021 to +0.034 macro-F1 (95% paired bootstrap CI, deduplicated test).

**Review queue (validation).** Each model sends `confidence < threshold` to a human. The thresholds were chosen on
validation so both models give staff the **same workload** (rule written before computing, see `docs/api.md`):

| Model | Threshold | Sent to review | Its errors caught | Misrouted per 1,000 tickets |
|---|---|---|---|---|
| LR | 0.60 | 14.6% | 44.5% | 46.8 |
| BERT | 0.83 | 14.5% | 55.4% | **34.2** (27% fewer) |

## Architecture

```
                        TRAINING (dvc repro, on a machine with the data)
 data/raw/CFPB.csv ─► ingest ─► validate ─► labels ─► split ─► train ─► evaluate (gate vs champion)
   (DVC, DagsHub)      junk rules  schema    renames    by date   TF-IDF+LR   macro-F1, small classes
                                                                     │            └─► MLflow (DagsHub)
                                                                     ▼
                         frozen models/ + manifest.json (sha256 of every file)

 git push ─► CI (GitHub Actions, no secrets): pytest on a synthetic fixture: pipeline smoke, goldens, API
                 │ green on main
                 ▼
             CD: dvc pull models ─► verify sha256 vs manifest ─► docker build (models baked in) ─► ghcr.io
                 │
                 ▼
             Render (free): the image ─► FastAPI ─► checksums re-verified at startup (mismatch = won't start)
                                            │
             POST /predict ─► 413 / no_content / too_short ─► model ─► threshold ─► auto_route | review
                                            │
                                            └─► Neon Postgres: one row per request (hash + length, NO text)
```

## Deployment decision (Day 13 rule, checked on Render on Day 20)

**Rule, written before any measurement:** deploy BERT if **p95 < 200 ms and memory ≤ 512 MB** on Render's free
instance; otherwise deploy LR and record BERT's numbers as evidence.

**What Render showed:** from my laptop, even `/health` (no work at all) takes ~270 ms, so **no model can pass
200 ms end-to-end**: the network alone fails it. The rule is therefore applied to **service time** (the app's own
`latency_ms`, stored in Neon), with end-to-end reported alongside.

| On Render free (100 val complaints) | Service p95 | End-to-end p95 | Peak memory | Rule |
|---|---|---|---|---|
| LR | 2.8 ms | 309 ms | TBD | TBD |
| BERT | TBD | TBD | TBD | TBD |

**Decision:** TBD. *(Local preview at `--cpus 0.1 --memory 512m`, mimicking the free tier: BERT p95 ≈ 2.4 s.)*

## Setup

```bash
git clone https://github.com/suvignan/Ticket-router && cd Ticket-router
python -m venv nlp && nlp\Scripts\activate            # Python 3.13
pip install -r requirements-train.txt                  # pipeline + tests (no TensorFlow)
pytest -q                                              # runs WITHOUT any data or secrets (synthetic fixture)
```

| File | For |
|---|---|
| `requirements-serve.txt` | The Docker image: API only (no pandas, no TensorFlow) |
| `requirements-train.txt` | + pipeline, DVC, MLflow, tests |
| `requirements-dev.txt` | + TensorFlow / KerasHub for the notebooks |

To get the data and models, add DagsHub credentials locally (never committed):

```bash
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <dagshub-user>
dvc remote modify origin --local password <token>
dvc pull
```

Run the API locally: `uvicorn app.main:app`, with `DATABASE_URL` set to store prediction rows.
Or in Docker: `docker build -t ticket-router . && docker run -p 8000:8000 -e DATABASE_URL ticket-router`.

## How to retrain

```bash
dvc pull            # raw data + champion models
dvc repro           # re-runs only the stages whose inputs changed
```

The `evaluate` stage is a **gate**: the new model must reach macro-F1 ≥ 0.65, may not drop more than 0.005 vs the
champion, and no small class may drop more than 0.02 (all on validation). A failed gate stops the run, and both
outcomes are logged to MLflow on DagsHub. Promoting a passing model to `models/` is a deliberate step: copy the
files, update `models/manifest.json`, `dvc add` + `dvc push`, then commit. CD then builds and deploys exactly those
files.

## Known limitations

- **Label noise.** The consumer picks the CFPB product, not a bank expert, so some "errors" are wrong labels.
  Credit reporting and debt collection overlap the most.
- **Drift.** Macro-F1 falls from val to test (LR 0.696 → 0.656, BERT 0.721 → 0.686) only 1–3 months later. The review
  shares chosen on validation drift too: on test, LR reviews 8.8% and BERT 19.0% (both 14.5% on val). Production
  needs monitoring of the review share and a retraining trigger. Today that's manual.
- **Company-name reliance.** Narratives name the company (when not redacted), and the model partly learns
  "company X → product Y". That won't transfer to one bank, where every ticket names the same company.
- **CFPB is not one bank.** US consumers writing to a regulator, about hundreds of companies, with `XXXX`
  redactions. A single Australian bank's tickets differ in teams, vocabulary, length and channel. This is a
  proof of method; it needs retraining on the bank's own labelled tickets.
- **BERT reads only the first 350 words** (~90% of complaints in full). The response says when (`truncated_for_model`).
- **Small classes** (e.g. prepaid card, vehicle loan) have far fewer examples and lower F1; macro-F1 averages them in.
- **Free-tier hosting:** the instance sleeps after 15 min idle (cold start), with a fraction of a CPU and 512 MB.
- **No authentication or rate limiting** on the public endpoint. The 35,000-character limit is the only protection.
- **CI never loads the real model files** (it has no secrets), so a library upgrade that changes the frozen models'
  outputs would pass CI. The fix: a reference-row check against the built image in CD.

## Notebooks

| # | Notebook | What it does |
|---|---|---|
| 01 | `01_data_loading.ipynb` | Load the raw CSV, keep 4 columns, run sanity checks (bad lines, duplicate IDs) |
| 02 | `02_eda.ipynb` | Class imbalance, volume over time, duplicates and template complaints |
| 03 | `03_labels.ipynb` | Merge renamed CFPB products, split the old card label, drop junk rows |
| 04 | `04_split.ipynb` | Time-based split: train 2023–2024, val Jan–Feb 2025, test Mar–May 2025 |
| 05 | `05_baseline.ipynb` | Text cleaning, TF-IDF + Logistic Regression baseline |
| 06 | `06_lstm.ipynb` | LSTM vs LR on truncated text |
| 07 / 07b | `07_transformer.ipynb`, `07b_bert.ipynb` | Fine-tune BERT (trained on Colab) |
| 7c | `7c_bert_onnx_export.ipynb` | Export BERT to ONNX for faster CPU serving |
| 08 | `08_test.ipynb` | One-time final test of all models + decision record |
| – | `memory_check.ipynb` | Peak memory check for serving |

## Incident log

- 2026-11-05: a DagsHub access token was exposed (shared via an editor selection in a chat tool). It was revoked
  and replaced. Credentials live only in git-ignored `.dvc/config.local` and `.env` locally, and in GitHub
  Actions secrets / Render environment variables in the cloud.
