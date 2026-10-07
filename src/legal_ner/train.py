"""Model loading, TrainingArguments, Trainer, and Drive checkpointing (spec section 7).

Tuned for Colab's free T4 (16 GB). Checkpoints are written under PROJECT_DIR (Drive) so
a disconnected session can resume. Resume logic is explicit (section 7): if a checkpoint
already exists in the output dir, training resumes from it.

Heavy imports (torch/transformers/datasets) are lazy so this module can be imported for
inspection without the full stack installed.

Version coupling: `eval_strategy` is correct for transformers==4.46.0 (the pinned
version). Do not rename it without the migration pass in spec section 11.
"""

from __future__ import annotations

import os

from . import config, labels as lbl
from .evaluate import make_compute_metrics


def load_model_and_tokenizer(model_name: str = config.MODEL_NAME):
    """Load the token-classification model + tokenizer with the 83-label head."""
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    label2id, id2label = lbl.build_label_maps()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # The checkpoint is -uncased; confirm the tokenizer lowercases (section 7 note).
    if not getattr(tokenizer, "do_lower_case", True):
        print(
            "WARNING: tokenizer is not lower-casing input, but the model is -uncased. "
            "Case-sensitive entity distinctions will be lost (see README limitations)."
        )

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
        eval_strategy="epoch",          # transformers>=4.46 spelling
        save_strategy="epoch",
        save_total_limit=2,             # avoid filling the Drive quota
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,  # effective batch size 32
        num_train_epochs=5,
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


def train(
    tokenized_train,
    tokenized_val,
    model_name: str = config.MODEL_NAME,
    output_dir: str | None = None,
):
    """Fine-tune the model, resuming from a Drive checkpoint if one exists.

    Persists labels.json alongside the checkpoints so inference never guesses label
    ordering (section 4.1). Returns the fitted Trainer.
    """
    from transformers import DataCollatorForTokenClassification, Trainer, set_seed

    set_seed(config.SEED)
    output_dir = output_dir or config.project_path(config.CHECKPOINTS_DIRNAME)
    os.makedirs(output_dir, exist_ok=True)

    # labels.json must travel with every checkpoint.
    lbl.save_labels(os.path.join(output_dir, config.LABELS_FILENAME))

    model, tokenizer = load_model_and_tokenizer(model_name)
    _, id2label = lbl.build_label_maps()

    args = build_training_args(output_dir)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(id2label),
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
