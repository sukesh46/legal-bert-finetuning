# CUAD → Legal-BERT Fine-Tuning Pipeline

Fine-tunes [`nlpaueb/legal-bert-base-uncased`](https://huggingface.co/nlpaueb/legal-bert-base-uncased)
for clause-level NER on the [CUAD](https://github.com/TheAtticusProject/cuad) dataset.
CUAD ships as SQuAD 2.0 question-answering; this pipeline converts it to token-level BIO
tags (83 labels), fine-tunes with the Hugging Face `Trainer`, evaluates entity-level F1
with `seqeval`, and runs inference that returns clause spans with character offsets back
into the original document.

## Architecture

Logic lives in an importable, unit-tested package (`src/legal_ner/`); the Colab notebook
is a thin orchestrator that just calls each stage. This keeps the correctness-critical
data transforms testable in milliseconds offline, with Colab reserved for training.

```
src/legal_ner/
  config.py         Single source of truth: WINDOW_SIZE, CUAD_CATEGORIES, paths, seed,
                    split ratios, Strategy A overlap priority order.
  labels.py         83-label BIO schema; labels.json save/load.
  download_cuad.py  Fetch + validate CUAD_v1.json (never re-hosted; fetched at runtime).
  cuad_to_bio.py    Char-span -> word-span -> BIO (the correctness core).
  chunking.py       Word-level sliding window (384 / stride 128).
  tokenize_align.py Subword tokenization + -100 label masking.
  train.py          Model load, TrainingArguments, Trainer, Drive checkpoint + resume.
  evaluate.py       seqeval F1 + per-category report (worst-first, low-support flagged).
  inference.py      Checkpoint -> raw contract -> clause spans with original offsets.
tests/              Pure-logic tests + a tiny synthetic CUAD fixture (no GPU, no network).
notebooks/
  pipeline.ipynb    Thin Colab orchestrator.
```

## Compatibility policy (read before changing versions)

- **Runtime: Python 3.10 on Colab (T4 GPU).** Not 3.9 (`datasets`/`evaluate` are moving
  to a 3.10 floor) and not 3.11+ (the 2024 cohort was validated against 3.10).
- The pinned library set is the **late-2024 mutually-tested cohort**, chosen for
  interoperability rather than recency:

  | package | version |
  |---|---|
  | transformers | 4.46.0 |
  | datasets | 3.0.1 |
  | tokenizers | 0.20.1 |
  | huggingface_hub | 0.26.2 |
  | accelerate | 1.0.1 |
  | evaluate | 0.4.3 |
  | seqeval | 1.2.2 |
  | numpy | >=1.26,<2.0 |

- `torch` is intentionally **not** pinned: use Colab's CUDA-matched build.
- The **"restart the runtime once after install"** step in the notebook is mandatory, not
  optional — it ensures the pinned `numpy`/`tokenizers` are the ones imported.
- Do not upgrade casually. See the spec's §11 migration guide first (transformers 5.x
  dropped TF/Flax and renamed arguments; `eval_strategy` is correct only for ≥4.46,<5).

## How to run (Colab)

Open `notebooks/pipeline.ipynb` in Colab (File → Open notebook → GitHub →
`sukesh46/legal-bert-finetuning` → `notebooks/pipeline.ipynb`) and set the hardware
accelerator to **T4 GPU** (Runtime → Change runtime type).

Stock Colab now ships **Python 3.13**, but the pinned cohort targets **3.10**. The
notebook obtains 3.10 via `condacolab` (pinned to `0.1.8`, whose Miniconda is built for
3.10), so the run order has a one-time kernel restart baked in:

1. **Cell A** installs `condacolab` and auto-restarts the kernel (expected, not a crash).
2. **Cell B** (after the restart) installs the pinned cohort into the conda 3.10 `base`.
3. **Cell C** verifies the GPU + versions.
4. Remaining cells: mount Drive → clone/pull this public repo into Drive → download →
   convert → chunk/tokenize → train → evaluate → inference.

Each stage is one call into `legal_ner`; adjust `limit=` in the download cell for a quick
smoke run on a handful of contracts.

> **Note on the conda route.** Pinning Python 3.10 onto a 3.13 Colab VM via `condacolab`
> is the pragmatic way to honor the pinned stack today, but it depends on an older
> `condacolab` release and conda's base-env behavior, so it is more fragile than a native
> install. If Colab's Python keeps advancing, the durable alternative is to re-pin the
> whole cohort to a current (3.13-compatible) set — transformers 5.x, numpy 2.x — per the
> migration guide in the spec (§11), which requires re-verifying the `Trainer` surface.

### Approximate T4 runtimes (fp16)

| stage | time |
|---|---|
| CUAD download + conversion | < 5 min |
| Fine-tuning, 5 epochs | ~30–60 min |

Checkpoints are written to Drive (`PROJECT_DIR/checkpoints`), so a disconnected session
resumes automatically on the next `train.train(...)` call.

## Strategy A: overlap-priority order and rationale

BIO is single-label-per-token, but CUAD spans can overlap across clause categories. v1
uses **priority-ordered single-label collapse** (`config.OVERLAP_PRIORITY`): when two
category spans overlap at a token, the higher-priority category wins, and the override is
recorded for audit (`ConversionStats.overlap_overrides`).

Priority rationale: **rarer, more specific clauses outrank generic, high-frequency ones.**
Structural categories that appear in nearly every contract — `Parties` and `Document Name`
— are ranked lowest, so they don't swallow rarer, higher-value clauses (e.g. `Source Code
Escrow`, `Most Favored Nation`) when spans collide. The audit counter lets you quantify how
much information the collapse discards per category pair; if a critical pair shows heavy
loss, Strategy B (multi-label sigmoid heads, spec §4.2) is the documented upgrade path.

## Known limitations

- **Uncased model.** `legal-bert-base-uncased` lowercases input, so case-based distinctions
  (defined term `Agreement` vs generic "agreement") are lost. A cased legal checkpoint
  (e.g. `law-ai/InLegalBERT`) is the alternative if that matters.
- **Single-label collapse.** Strategy A discards the lower-priority span wherever categories
  overlap. Quantified by the overlap-override audit log.
- **Chunking boundary effects.** Entities straddling window boundaries are merged at
  inference by preferring a non-`O` prediction over `O` (earliest non-`O` window wins).
  Windowing is done at the word level with a conservative token budget; the tokenizer's
  `max_length` truncation is a hard safety net.

## Local development & testing

The deterministic modules (config, labels, cuad_to_bio, chunking, tokenize_align,
download validation, and the pure parts of evaluate/inference) are fully unit-tested and
need neither a GPU nor the heavy ML stack.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt   # pytest + numpy (+ pinned cohort)
.venv/bin/pytest
```

Tests run against `tests/fixtures/tiny_cuad.json`, a tiny synthetic CUAD file whose
character offsets are verified to slice correctly. The seqeval-backed per-category report
test is skipped automatically if `seqeval` isn't installed locally; it runs on Colab.

## Acceptance criteria (from the spec)

1. Span reconstruction passes for the sampled examples — enforced in `cuad_to_bio` tests
   and asserted in the notebook's conversion cell.
2. No contract appears in more than one split — `contract_level_split` + its leakage test.
3. 5 epochs train without OOM on a T4 at the specified batch size.
4. `evaluate` produces a full per-category report, not just aggregate F1.
5. `inference` offsets, sliced from the original text, correspond to real clauses.

## Data & license

CUAD is distributed under CC BY 4.0 by The Atticus Project. This pipeline fetches
`CUAD_v1.json` at runtime and never re-hosts or bundles the raw file.
