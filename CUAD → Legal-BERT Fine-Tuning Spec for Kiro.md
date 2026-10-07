# Engineering Spec: CUAD → BIO Fine-Tuning Pipeline for Legal-BERT (Google Colab)

**Audience**: Kiro (AI coding agent) **Objective**: Produce a complete, runnable Colab notebook (or a set of `.py` modules importable from a notebook) that fine-tunes `nlpaueb/legal-bert-base-uncased` for clause-level NER/span-classification on the CUAD dataset, evaluates with entity-level F1 via `seqeval`, and persists checkpoints to Google Drive.

---

## 0. Deliverables checklist

- [ ] `setup.ipynb` or first notebook cell block — environment setup (Python 3.10 + pinned 2024 cohort), GPU check, Drive mount, mandatory one-time runtime restart, and `requirements.lock.txt` capture
- [ ] `download_cuad.py` — fetches and validates the CUAD dataset
- [ ] `cuad_to_bio.py` — converts CUAD's SQuAD-style span annotations into token-level BIO tags
- [ ] `chunking.py` — splits long contracts into model-length-safe, overlapping windows while preserving label alignment
- [ ] `tokenize_align.py` — subword tokenization + label alignment (reuses the `-100` masking convention)
- [ ] `train.py` — model loading, `TrainingArguments`, `Trainer`, checkpointing to Drive
- [ ] `evaluate.py` — entity-level F1/precision/recall via `seqeval`, confusion analysis on top-confused clause types
- [ ] `inference.py` — loads a saved checkpoint, runs on a raw contract text file, outputs highlighted clause spans
- [ ] `README.md` — how to run each stage, expected runtimes on Colab T4

---

## 1. Environment setup (Colab specifics)

> **Compatibility policy (read first).** This pipeline deliberately targets a **conservative, mutually-tested library cohort from late 2024**, not the latest releases. Rationale: the whole stack (`transformers` 4.46 ↔ `tokenizers` ↔ `huggingface_hub` ↔ `datasets` 3.0 ↔ `accelerate` 1.0) shipped in the same window and is known to interoperate. The latest `transformers` is 5.x, which **dropped TensorFlow/Flax, removed several pipelines, and renamed arguments** — adopting it would mean debugging a library migration at the same time as the data-conversion logic. Do **not** bump these pins without a deliberate migration pass (see §11). The pins below still resolve on PyPI today.

### 1.0 Runtime: Python 3.10

- **Select Python 3.10** for the Colab runtime (not 3.9, not 3.11+).
  - Why not 3.9: `datasets`/`evaluate` are drifting toward a 3.10 floor (`datasets` 5.0 already requires ≥3.10). 3.9 ages out first and buys nothing here — the 4.46-era stack runs identically on 3.9 and 3.10.
  - Why not 3.11+: unnecessary; the 2024 cohort was validated against 3.10, and newer interpreters increase the chance of a preinstalled-wheel mismatch.
  - Verify at the top of the notebook:
    ```python
    import sys
    assert sys.version_info[:2] == (3, 10), f"Expected Python 3.10, got {sys.version.split()[0]}"
    ```

1. Runtime: **Runtime → Change runtime type → T4 GPU** (free tier). Verify in code:

   ```python
   import torch
   assert torch.cuda.is_available(), "GPU not enabled — check Runtime settings"
   print(torch.cuda.get_device_name(0))
   ```
2. Mount Google Drive immediately — Colab's local disk is ephemeral and wiped on disconnect. All checkpoints, the converted dataset, and logs must live under a Drive path, e.g. `/content/drive/MyDrive/legal_ner_project/`.

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   PROJECT_DIR = '/content/drive/MyDrive/legal_ner_project'
   ```
3. Install the pinned cohort. **Pin everything that affects loading/training**, including `torch`, `numpy`, `tokenizers`, and `huggingface_hub` — leaving these floating is the actual source of "it worked yesterday" Colab breakage, because Colab's preinstalled `torch`/`numpy` drift independently of what you `pip install`.

   ```bash
   !pip install -q \
       transformers==4.46.0 \
       datasets==3.0.1 \
       tokenizers==0.20.1 \
       huggingface_hub==0.26.2 \
       accelerate==1.0.1 \
       evaluate==0.4.3 \
       seqeval==1.2.2 \
       "numpy>=1.26,<2.0"
   ```

   Notes:
   - `transformers==4.46.0` requires Python ≥3.8 and is the version where the `TrainingArguments` eval argument is spelled **`eval_strategy`** (older versions used `evaluation_strategy`; v5 is different again). The code in §7 assumes this exact spelling.
   - `numpy` is held below 2.0: the 2024 torch/transformers wheels were built against the NumPy 1.26 ABI, and letting NumPy 2.x install is a common source of `_ARRAY_API not found` import crashes on Colab.
   - `torch` is intentionally **not** pinned in the install line — use the CUDA-matched build Colab ships for the selected runtime, then assert the version at runtime rather than forcing a reinstall (reinstalling torch on Colab often breaks the CUDA driver match). Record the resolved `torch.__version__` in the run log.
   - After install, **restart the runtime once** (`Runtime → Restart session`) so the newly pinned `numpy`/`tokenizers` are the ones imported, then re-run from the top. This single restart avoids the most common Colab dependency-resolution failure.

4. Pin the resolved environment for reproducibility: after a successful install+restart, run `!pip freeze > {PROJECT_DIR}/requirements.lock.txt` so the exact resolved graph (including transitive deps) travels with the project.
5. Set and log a random seed everywhere (`transformers.set_seed(42)`) — required for reproducibility discussions later. Also log `sys.version`, `torch.__version__`, `transformers.__version__`, and `datasets.__version__` to the run log at startup.

---

## 2. Data acquisition (`download_cuad.py`)

- Source: `https://github.com/TheAtticusProject/cuad` (CC BY 4.0). The canonical file is `CUAD_v1.json`, SQuAD 2.0-formatted.
- Download via `!wget` or `requests`, verify file integrity by checking it parses as valid JSON and has the expected top-level key `"data"`.
- **Do not re-host or bundle the raw CUAD file in any public repo** — always fetch at runtime per CUAD's distribution terms; confirm license text in README.
- Log basic stats after download: number of contracts, number of QA pairs, number of unique clause-type categories (expect 41 categories + 1 implicit "no answer" case per SQuAD 2.0 convention).

## 3. Understanding CUAD's native format (document this clearly in code comments for Kiro's own sanity-check)

CUAD is structured as SQuAD 2.0 QA, **not** native NER:

```json
{
  "data": [
    {
      "title": "ContractName",
      "paragraphs": [
        {
          "context": "<full contract text>",
          "qas": [
            {
              "question": "Highlight the parts (if any) ... related to \"Governing Law\"...",
              "id": "...",
              "answers": [{"text": "This Agreement shall be governed by...", "answer_start": 4821}],
              "is_impossible": false
            },
            ... one qas entry per clause category (41 total) per contract ...
          ]
        }
      ]
    }
  ]
}
```

Key implications Kiro must handle:

- **Each contract has up to 41 separate span-extraction questions**, one per clause category, not a single multi-class tag sequence.
- **Spans can overlap** across categories (e.g., a sentence can simultaneously be tagged `Governing Law` and contain a `Document Name` reference). A single BIO sequence cannot represent overlapping multi-label spans natively — see §4.2 for the resolution strategy.
- `answer_start` is a **character offset** into `context`, not a token or word index — must be converted.
- `is_impossible: true` means that clause category does not appear in this contract at all — skip, do not treat as a negative span.

## 4. CUAD → BIO conversion (`cuad_to_bio.py`)

### 4.1 Target label schema

Define explicitly, do not infer dynamically — hardcode the 41 CUAD category names as a constant list, e.g.:

```python
CUAD_CATEGORIES = [
    "Document Name", "Parties", "Agreement Date", "Effective Date",
    "Expiration Date", "Renewal Term", "Notice Period To Terminate Renewal",
    "Governing Law", "Most Favored Nation", "Non-Compete",
    "Exclusivity", "No-Solicit Of Customers", "Competitive Restriction Exception",
    "No-Solicit Of Employees", "Non-Disparagement", "Termination For Convenience",
    "Rofr/Rofo/Rofn", "Change Of Control", "Anti-Assignment",
    "Revenue/Profit Sharing", "Price Restrictions", "Minimum Commitment",
    "Volume Restriction", "IP Ownership Assignment", "Joint IP Ownership",
    "License Grant", "Non-Transferable License", "Affiliate License-Licensor",
    "Affiliate License-Licensee", "Unlimited/All-You-Can-Eat License",
    "Irrevocable Or Perpetual License", "Source Code Escrow",
    "Post-Termination Services", "Audit Rights", "Uncapped Liability",
    "Cap On Liability", "Liquidated Damages", "Warranty Duration",
    "Insurance", "Covenant Not To Sue", "Third Party Beneficiary"
]
assert len(CUAD_CATEGORIES) == 41
```

Build label list as BIO: `["O"] + [f"{p}-{c}" for c in CUAD_CATEGORIES for p in ("B", "I")]` → 83 labels total. Build `label2id` / `id2label` dicts and persist them to a `labels.json` in `PROJECT_DIR` — this file must travel with every checkpoint so inference never guesses label ordering.

### 4.2 Resolving overlapping multi-label spans

Because BIO is single-label-per-token, Kiro must implement **one** of these strategies — pick the first for v1 (simplicity), document the limitation explicitly in README:

- **Strategy A (v1, recommended)**: Priority-ordered single-label collapse. Define a fixed priority order over the 41 categories (e.g., more specific/rare categories outrank generic ones — document the chosen order and rationale). When two spans overlap at a token, the higher-priority category's tag wins. Log a counter of how many tokens were affected by an overlap resolution, per category pair, so you can audit how much information is being discarded.
- **Strategy B (v2/stretch)**: Multi-label token classification — replace the single softmax classification head with 41 independent sigmoid heads (one per category, each predicting B/I/O independently). This fully preserves overlaps but requires a custom model class subclassing `AutoModelForTokenClassification`'s base and a custom multi-label loss (`BCEWithLogitsLoss` per category). Only implement this if Strategy A's audit log shows overlap-driven information loss is unacceptable for the target categories in your specific use case (e.g., governing-law + liability overlaps are rare, but IP-ownership + license-grant overlaps may not be).

### 4.3 Char-span → word-span → BIO conversion algorithm

For each `(context, qas)` pair:

1. Whitespace-tokenize `context` into words, recording each word's `(start_char, end_char)` offset. Use a simple regex-based tokenizer (`\S+` matches) rather than `.split()` alone, since `.split()` discards offset information.
2. Initialize a per-contract tag array of length `len(words)`, all `"O"`.
3. For each non-impossible `qas` entry: a. Extract the category name from the `question` string (CUAD's questions follow a fixed template — Kiro must write a regex or exact-match lookup against `CUAD_CATEGORIES` to extract which category this `qas` entry corresponds to; do not assume ordering is stable across contracts). b. For each `answer` in `answers` (CUAD sometimes gives multiple valid spans per category per contract):
   - Find all words whose `(start_char, end_char)` overlaps the answer's `(answer_start, answer_start + len(answer_text))`.
   - Tag the first overlapping word `B-{category}`, subsequent overlapping words `I-{category}`, **applying the Strategy A priority rule** if a word is already tagged by a higher-priority category.
4. Store the result as a row: `{"contract_id": ..., "words": [...], "ner_tags": [...]}` — same shape as the CoNLL example from the earlier walkthrough, so downstream tokenization/alignment code is reusable unchanged.
5. **Validation step (mandatory)**: for a random sample of \~20 converted examples, reconstruct the tagged spans back into text and print them next to the original `answer` text from CUAD — they must match exactly. This is the single most important correctness check in this entire pipeline; a silent off-by-one in char→word mapping will silently corrupt every downstream label.

### 4.4 Output format

Save the converted dataset as a Hugging Face `datasets.Dataset` (via `Dataset.from_list(...)`), split 80/10/10 train/val/test **at the contract level** (not paragraph level) to prevent leakage — paragraphs from the same contract must never appear in both train and test. Save to `PROJECT_DIR/cuad_bio_dataset/` using `.save_to_disk()`.

## 5. Chunking long contracts (`chunking.py`)

CUAD contracts routinely exceed Legal-BERT's 512-token limit (your stated use case: documents up to 1000 pages — CUAD contracts are shorter but still frequently truncation-risk at 512 tokens).

- Implement a **sliding window with stride**: window size 384 tokens, stride 128 tokens (i.e., 256-token overlap) — standard values for long-document NER, balances context preservation against redundant compute.
- Critical requirement: splitting must happen at the **word level before tokenization**, not mid-word, and BIO tags must travel with their words into each window.
- Edge case Kiro must handle: an entity span that starts in the overlap region of window N and window N+1 will appear (partially or fully) in both windows. At inference-aggregation time, deduplicate by preferring the window where the entity's full `B-...` to end-of-`I-...` span is most complete; document this merge rule clearly in `inference.py`.
- Output: each chunk becomes its own training example with its own `words`/`ner_tags`, tagged with a `(contract_id, chunk_index)` for traceability.

## 6. Tokenization + label alignment (`tokenize_align.py`)

Reuse the exact alignment convention from the CoNLL walkthrough, generalized to the 83-label CUAD schema:

```python
def tokenize_and_align_labels(examples, tokenizer, label2id):
    tokenized = tokenizer(examples["words"], is_split_into_words=True,
                           truncation=True, max_length=384)
    all_labels = []
    for i, tags in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_id = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                label_ids.append(label2id[tags[word_id]])
            else:
                label_ids.append(-100)
            previous_word_id = word_id
        all_labels.append(label_ids)
    tokenized["labels"] = all_labels
    return tokenized
```

Note: `max_length=384` here matches the chunking window size from §5 — these two numbers must stay in sync; define both from a single shared constant (`WINDOW_SIZE = 384`) rather than hardcoding twice.

## 7. Model & training (`train.py`)

- Base model: `nlpaueb/legal-bert-base-uncased`. Load with:

  ```python
  model = AutoModelForTokenClassification.from_pretrained(
      "nlpaueb/legal-bert-base-uncased",
      num_labels=83, id2label=id2label, label2id=label2id
  )
  ```
- **Case sensitivity note**: this checkpoint is `-uncased` — verify the tokenizer lowercases input (`tokenizer.do_lower_case == True`). If case-sensitive entity distinctions matter later (e.g., distinguishing a defined term `Agreement` from generic "agreement"), flag this as a known limitation in README; `law-ai/InLegalBERT` or a cased legal checkpoint would be the alternative.
- Training arguments, tuned for Colab's free T4 (16GB VRAM). **Version coupling:** the eval argument below is spelled `eval_strategy`, which is correct for `transformers==4.46.0` (the pinned version). On transformers <4.46 it is `evaluation_strategy`; on transformers 5.x the Trainer surface changed again. Do not change this spelling unless you also migrate the pin (see §11).

  ```python
  TrainingArguments(
      output_dir=f"{PROJECT_DIR}/checkpoints",
      eval_strategy="epoch",
      save_strategy="epoch",
      save_total_limit=2,                 # avoid filling Drive quota
      learning_rate=2e-5,
      per_device_train_batch_size=16,
      per_device_eval_batch_size=16,
      gradient_accumulation_steps=2,      # effective batch size 32
      num_train_epochs=5,
      weight_decay=0.01,
      fp16=True,                          # mixed precision — ~2x speedup on T4
      logging_steps=50,
      load_best_model_at_end=True,
      metric_for_best_model="f1",
      report_to="none",
  )
  ```
- **Colab-specific resilience requirement**: free-tier Colab sessions can disconnect mid-training. `save_strategy="epoch"` + saving to Drive (not local `/content`) means a disconnected run can be resumed via `Trainer(..., resume_from_checkpoint=True)` pointed at the last saved checkpoint in Drive. Kiro must write this resume logic explicitly, not assume the user will figure it out.
- Class imbalance warning: `"O"` will vastly outnumber any entity tag (most contract text isn't a tagged clause). Do not apply class weighting by default — try without first, since over-correcting for imbalance in NER often hurts precision; only add weighted loss if the baseline run shows a specific category being systematically missed.

## 8. Evaluation (`evaluate.py`)

- Use the same `seqeval`-based `compute_metrics` pattern from the CoNLL walkthrough, generalized to 83 labels.
- **Required additional output**: per-category F1 breakdown (`seqeval.classification_report(true_labels, true_preds)`), not just the aggregate — with 41 categories of wildly varying frequency (e.g., "Parties" appears in nearly every contract; "Source Code Escrow" is rare), the overall F1 will mask categories that are performing poorly. Print this as a sorted table, worst-performing categories first.
- Flag any category with fewer than \~20 training examples after the 80/10/10 split as "insufficient data — F1 on this category is not statistically meaningful" rather than letting it silently appear as a 0.0 or 1.0 F1 outlier.

## 9. Inference (`inference.py`)

- Load `labels.json` + the saved checkpoint.
- Input: raw contract text (string or `.txt`/`.docx` path).
- Pipeline: whitespace-tokenize with offsets (reuse §4.3's tokenizer) → chunk per §5 → tokenize+predict per chunk → merge overlapping-window predictions per §5's dedup rule → reconstruct contiguous entity spans from the BIO output → return a list of `{category, text, start_char, end_char}`.
- Output format must carry character offsets back to the *original* document (not just the chunk), since this is what downstream citation-grounding (page/section linking) depends on.

## 10. README requirements

Must document: how to run each script in order, expected Colab T4 runtime per stage (rough estimate: full CUAD conversion \<5 min; full fine-tuning 5 epochs \~30–60 min on T4 with fp16), the Strategy A overlap-priority order chosen and why, known limitations (uncased model, single-label collapse, chunking boundary effects), and exact pinned library versions.

The README must also state the **compatibility policy** explicitly:

- Target runtime: **Python 3.10** on Colab (T4 GPU).
- The pinned cohort is the **late-2024 mutually-tested set** (`transformers==4.46.0`, `datasets==3.0.1`, `tokenizers==0.20.1`, `huggingface_hub==0.26.2`, `accelerate==1.0.1`, `evaluate==0.4.3`, `seqeval==1.2.2`, `numpy<2.0`). These are chosen for interoperability, not recency.
- Reproduce the exact environment from `requirements.lock.txt` (committed after the first successful run).
- The "restart runtime once after install" step is mandatory, not optional — document it prominently.
- Point readers to §11 before they attempt any version upgrade.

## 11. Version-upgrade / migration guide (do not upgrade casually)

This pipeline is pinned on purpose. If a future maintainer needs to move to the current `transformers` 5.x line, treat it as a deliberate migration, not a `pip install -U`:

- **`transformers` 4.46 → 5.x breaking changes to expect:** TensorFlow and Flax support dropped (irrelevant here, PyTorch-only), several pipelines removed, and a batch of renamed arguments. Re-verify the `Trainer` / `TrainingArguments` surface used in §7, especially `eval_strategy`, `save_strategy`, `load_best_model_at_end`, and `fp16`.
- **`datasets` 3.0 → 5.0:** requires Python ≥3.10 (already satisfied here) but has its own loading/`save_to_disk` behavior changes — re-run the §4.3.5 span-reconstruction validation after any `datasets` bump, since a serialization change there would silently corrupt labels.
- **`numpy` 1.x → 2.x:** only after confirming the resolved `torch` build is compiled against the NumPy 2.x ABI; otherwise expect import-time crashes.
- **Migration gate:** before accepting any upgrade, the full acceptance-criteria list below must pass again on the new stack — in particular criterion 1 (span reconstruction) and criterion 3 (5 epochs, no OOM on T4). Record the new resolved versions in a fresh `requirements.lock.txt`.

---

## Acceptance criteria (how to know this is done correctly)

1. `cuad_to_bio.py`'s validation step (§4.3.5) passes for 20/20 sampled examples with exact span reconstruction.
2. No contract's paragraphs appear in more than one of train/val/test.
3. Training completes 5 epochs without OOM on a T4 GPU using the batch size specified.
4. `evaluate.py` produces a full per-category report, not just aggregate F1.
5. `inference.py` run on a held-out contract produces spans whose character offsets, when sliced from the original text, visually correspond to real clauses on manual spot-check.