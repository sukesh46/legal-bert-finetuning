"""Single source of truth for pipeline constants.

Every other module imports its constants from here. In particular, WINDOW_SIZE is
used by BOTH chunking (section 5) and tokenization max_length (section 6); the spec
requires these stay in sync, so they must come from this one definition and never be
hardcoded twice.
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# On Colab this should point under the mounted Drive, e.g.
#   /content/drive/MyDrive/legal_ner_project
# Overridable via the LEGAL_NER_PROJECT_DIR env var so local dev and Colab can
# differ without editing code. Locally it defaults to ./project_dir.
PROJECT_DIR = os.environ.get(
    "LEGAL_NER_PROJECT_DIR",
    os.path.join(os.getcwd(), "project_dir"),
)

# Derived sub-paths (kept as functions of PROJECT_DIR so a single override cascades).
def project_path(*parts: str) -> str:
    """Join path parts under PROJECT_DIR."""
    return os.path.join(PROJECT_DIR, *parts)


CUAD_JSON_FILENAME = "CUAD_v1.json"
LABELS_FILENAME = "labels.json"
DATASET_DIRNAME = "cuad_bio_dataset"
CHECKPOINTS_DIRNAME = "checkpoints"

# --------------------------------------------------------------------------- #
# Chunking / tokenization window (section 5 + section 6 MUST share this)
# --------------------------------------------------------------------------- #
WINDOW_SIZE = 384          # sliding-window size in tokens; also tokenizer max_length
STRIDE = 128               # step between windows -> 256-token overlap
assert 0 < STRIDE < WINDOW_SIZE, "STRIDE must be a positive value smaller than WINDOW_SIZE"

# --------------------------------------------------------------------------- #
# Dataset splitting (section 4.4) — contract-level, not paragraph-level
# --------------------------------------------------------------------------- #
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9, "Split ratios must sum to 1.0"

# A category with fewer than this many training examples after the split is flagged
# as statistically meaningless rather than reported as a 0.0/1.0 F1 outlier (section 8).
MIN_CATEGORY_SUPPORT = 20

# --------------------------------------------------------------------------- #
# CUAD clause categories (section 4.1) — hardcoded, exactly 41, order is canonical.
# --------------------------------------------------------------------------- #
CUAD_CATEGORIES: list[str] = [
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
    "Insurance", "Covenant Not To Sue", "Third Party Beneficiary",
]
assert len(CUAD_CATEGORIES) == 41, f"Expected 41 CUAD categories, got {len(CUAD_CATEGORIES)}"
assert len(set(CUAD_CATEGORIES)) == 41, "CUAD categories must be unique"

# --------------------------------------------------------------------------- #
# Strategy A: overlap-resolution priority order (section 4.2)
# --------------------------------------------------------------------------- #
# BIO is single-label-per-token. When two category spans overlap at a token, the
# higher-priority (lower index in this list) category's tag wins. The rationale
# (documented in README): rarer / more specific clause types outrank generic,
# high-frequency ones, so that frequent categories like "Parties" / "Document Name"
# do not swallow rarer, higher-value clauses when spans collide.
#
# This list is a permutation of CUAD_CATEGORIES: specific/rare first, generic last.
OVERLAP_PRIORITY: list[str] = [
    # Rare, high-value, narrowly-scoped clauses first
    "Source Code Escrow", "Most Favored Nation", "Joint IP Ownership",
    "Unlimited/All-You-Can-Eat License", "Irrevocable Or Perpetual License",
    "Non-Transferable License", "Affiliate License-Licensor", "Affiliate License-Licensee",
    "Covenant Not To Sue", "Third Party Beneficiary", "Liquidated Damages",
    "Uncapped Liability", "Cap On Liability", "Volume Restriction",
    "Minimum Commitment", "Revenue/Profit Sharing", "Price Restrictions",
    "Competitive Restriction Exception", "No-Solicit Of Employees",
    "No-Solicit Of Customers", "Non-Disparagement", "Non-Compete", "Exclusivity",
    "Rofr/Rofo/Rofn", "Change Of Control", "Anti-Assignment",
    "IP Ownership Assignment", "License Grant", "Post-Termination Services",
    "Audit Rights", "Warranty Duration", "Insurance",
    "Notice Period To Terminate Renewal", "Renewal Term",
    "Termination For Convenience", "Expiration Date", "Effective Date",
    "Agreement Date", "Governing Law",
    # Generic, high-frequency, structural clauses last
    "Parties", "Document Name",
]
assert sorted(OVERLAP_PRIORITY) == sorted(CUAD_CATEGORIES), (
    "OVERLAP_PRIORITY must be a permutation of CUAD_CATEGORIES"
)

# Fast lookup: category -> priority rank (0 = highest priority / wins ties).
PRIORITY_RANK: dict[str, int] = {cat: i for i, cat in enumerate(OVERLAP_PRIORITY)}
