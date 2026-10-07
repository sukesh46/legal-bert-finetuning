"""Model loading, TrainingArguments, Trainer, and Drive checkpointing (spec section 7).

Tuned for Colab's free T4 (16 GB). Checkpoints are written under PROJECT_DIR (Drive) so
a disconnected session can resume. Resume logic is explicit (section 7): if a checkpoint
already exists in the output dir, training resumes from it.

Heavy imports (torch/transformers/datasets) are lazy so this module can be imported for
inspection without the full stack installed.

Version coupling (transformers 5.x): `eval_strategy` is the correct TrainingArguments
spelling, and `Trainer` takes `processing_class=` (the old `tokenizer=` was removed in
v5). Do not change these without the migration pass in spec section 11.
"""

from __future__ import annotations

import os

from . import config, labels as lbl
from .evaluate import make_compute_metrics


def load_tokenizer(model_name: str = config.MODEL_NAME):
    """Load the tokenizer and warn if it won't lowercase for an -uncased checkpoint."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not getattr(tokenizer, "do_lower_case", True):
        print(
            "WARNING: tokenizer is not lower-casing input, but the model is -uncased. "
            "Case-sensitive entity distinctions will be lost (see README limitations)."
        )
    return tokenizer


def load_model_and_tokenizer(model_name: str = config.MODEL_NAME):
    """Load the (Strategy A) token-classification model + tokenizer with the 83-label head."""
    from transformers import AutoModelForTokenClassification

    label2id, id2label = lbl.build_label_maps()
    tokenizer = load_tokenizer(model_name)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=lbl.num_labels(),
        id2label=id2label,
        label2id=label2id,
    )
    return model, tokenizer


def build_training_args(output_dir: str | None = None):
    """TrainingArguments tuned for a free-tier T4 (section 7)."""
    from transformers import TrainingArguments

    output_dir = output_dir or config.project_path(config.CHECKPOINTS_DIRNAME)
    return TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",          # transformers 5.x spelling (renamed from 4.46)
        save_strategy="epoch",
        save_total_limit=2,             # avoid filling the Drive quota
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,  # effective batch size 32
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        weight_decay=0.01,
        fp16=True,                      # mixed precision — ~2x speedup on T4
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        seed=config.SEED,
    )


def _has_resumable_checkpoint(output_dir: str) -> bool:
    """True if output_dir already contains a 'checkpoint-*' directory to resume from."""
    if not os.path.isdir(output_dir):
        return False
    return any(
        name.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, name))
        for name in os.listdir(output_dir)
    )


def _multilabel_data_collator(tokenizer, num_categories: int):
    """Collator for Strategy B: pads input fields and 2D (seq, C) label rows.

    HF's DataCollatorForTokenClassification only handles 1D integer label sequences, so
    for Strategy B we pad manually: inputs via the tokenizer, and each example's label
    matrix with all--100 rows up to the batch's max length.
    """
    import torch

    def collate(features):
        labels = [f["labels"] for f in features]
        inputs = [{k: v for k, v in f.items() if k != "labels"} for f in features]
        batch = tokenizer.pad(inputs, padding=True, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]

        ignore_row = [-100.0] * num_categories
        padded = []
        for row in labels:
            row = [list(r) for r in row]
            if len(row) < max_len:
                row = row + [list(ignore_row) for _ in range(max_len - len(row))]
            else:
                row = row[:max_len]
            padded.append(row)
        batch["labels"] = torch.tensor(padded, dtype=torch.float32)
        return batch

    return collate


def train(
    tokenized_train,
    tokenized_val,
    model_name: str = config.MODEL_NAME,
    output_dir: str | None = None,
):
    """Fine-tune the model, resuming from a Drive checkpoint if one exists.

    Dispatches on config.STRATEGY:
      * "A" — AutoModelForTokenClassification (83-label softmax) + seqeval metrics.
      * "B" — multi-label per-category model (model.py) + per-category BCE metrics.

    Persists labels.json alongside the checkpoints so inference never guesses label
    ordering (section 4.1). Returns the fitted Trainer.
    """
    from transformers import DataCollatorForTokenClassification, Trainer, set_seed

    set_seed(config.SEED)
    # Strategy A and B produce INCOMPATIBLE checkpoints (83-way softmax head vs 41-head
    # sigmoid). Keep them in separate subdirectories so resume_from_checkpoint never tries
    # to load one strategy's weights into the other's architecture.
    if output_dir is None:
        output_dir = config.project_path(
            config.CHECKPOINTS_DIRNAME, f"strategy_{config.STRATEGY.lower()}"
        )
    os.makedirs(output_dir, exist_ok=True)

    # labels.json must travel with every checkpoint (both strategies).
    lbl.save_labels(os.path.join(output_dir, config.LABELS_FILENAME))

    tokenizer = load_tokenizer(model_name)
    args = build_training_args(output_dir)

    if config.STRATEGY == "B":
        from .evaluate import make_compute_metrics_multilabel
        from .model import build_multilabel_model, strategy_b_categories

        categories = strategy_b_categories()
        model = build_multilabel_model(model_name, categories)
        data_collator = _multilabel_data_collator(tokenizer, len(categories))
        compute_metrics = make_compute_metrics_multilabel(categories)
    else:
        from transformers import AutoModelForTokenClassification

        label2id, id2label = lbl.build_label_maps()
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=lbl.num_labels(),
            id2label=id2label, label2id=label2id,
        )
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        compute_metrics = make_compute_metrics(id2label)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        # transformers 5.x removed Trainer(tokenizer=...); use processing_class.
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Colab resilience (section 7): resume from the last Drive checkpoint if present.
    resume = _has_resumable_checkpoint(output_dir)
    if resume:
        print(f"Resuming training from existing checkpoint in {output_dir}")
    trainer.train(resume_from_checkpoint=resume)

    # Save the final best model + tokenizer + labels next to the checkpoints.
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    lbl.save_labels(os.path.join(final_dir, config.LABELS_FILENAME))
    return trainer
