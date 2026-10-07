"""CUAD -> BIO fine-tuning pipeline for Legal-BERT (clause-level NER).

Public modules:
    config          Single source of truth for constants (window size, categories,
                    paths, seed, split ratios, overlap priority order).
    labels          BIO label-schema construction and labels.json persistence.
    download_cuad   Fetch + validate CUAD_v1.json.
    cuad_to_bio     SQuAD-style char spans -> token-level BIO tags (correctness core).
    chunking        Sliding-window splitting of long contracts (word level).
    tokenize_align  Subword tokenization + -100 label alignment.
    train           Model loading, TrainingArguments, Trainer, Drive checkpointing.
    evaluate        Entity-level F1 via seqeval + per-category report.
    inference       Checkpoint -> raw contract -> highlighted clause spans.
"""

__version__ = "0.1.0"
