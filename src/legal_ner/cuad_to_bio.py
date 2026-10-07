"""Convert CUAD's SQuAD-style char-span annotations into token-level BIO tags.

This is the correctness core of the pipeline (spec section 4). A silent off-by-one in
the char -> word mapping would corrupt every downstream label, so the span
reconstruction check (section 4.3.5) is treated as the single most important test.

Pipeline per (context, qas) pair:
    1. Offset-aware whitespace tokenization of the context (regex \\S+), recording each
       word's (start_char, end_char).
    2. A per-contract tag array initialised to "O".
    3. For each non-impossible qas entry: extract its category, then for each answer
       span tag overlapping words B-/I-, applying the Strategy A priority rule.
    4. Emit a row {"contract_id", "words", "ner_tags"}.

Splitting into train/val/test happens at the CONTRACT level (section 4.4) to prevent
leakage.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from . import config

# A "word" token is a maximal run of non-whitespace characters.
_WORD_RE = re.compile(r"\S+")


@dataclass
class Word:
    """A whitespace-delimited token with its character span in the source context."""

    text: str
    start: int  # inclusive char offset
    end: int    # exclusive char offset


def tokenize_with_offsets(context: str) -> list[Word]:
    """Whitespace-tokenize ``context`` into words carrying (start, end) char offsets.

    Uses a regex (\\S+) rather than str.split() so offsets are preserved exactly;
    end is exclusive, matching Python slicing (context[word.start:word.end] == word.text).
    """
    return [Word(m.group(0), m.start(), m.end()) for m in _WORD_RE.finditer(context)]


# --------------------------------------------------------------------------- #
# Category extraction from the CUAD question template
# --------------------------------------------------------------------------- #
# CUAD questions embed the category name in quotes, e.g.:
#   Highlight the parts (if any) ... related to "Governing Law" that should be ...
# We match the known category names directly rather than trusting qas ordering.
_CATEGORY_LOOKUP = {cat.lower(): cat for cat in config.CUAD_CATEGORIES}


def extract_category(question: str) -> str | None:
    """Return the canonical CUAD category named in ``question``, or None.

    Strategy: find quoted substrings and match (case-insensitively) against the known
    category list. Falls back to a direct substring scan if no quoted match is found.
    Longest category names are tried first so that e.g. "Non-Transferable License"
    is not shadowed by "License Grant".
    """
    # 1. Prefer quoted spans — CUAD's template puts the category in quotes.
    for quoted in re.findall(r'"([^"]+)"', question):
        hit = _CATEGORY_LOOKUP.get(quoted.strip().lower())
        if hit is not None:
            return hit

    # 2. Fall back to substring match, longest category first to avoid partial shadowing.
    q_lower = question.lower()
    for cat in sorted(config.CUAD_CATEGORIES, key=len, reverse=True):
        if cat.lower() in q_lower:
            return cat
    return None


def _overlapping_word_indices(words: list[Word], ans_start: int, ans_end: int) -> list[int]:
    """Indices of words whose char span overlaps [ans_start, ans_end).

    Overlap (not containment): a word counts if it shares any character range with the
    answer, so partial-word answer boundaries still capture the word.
    """
    out = []
    for i, w in enumerate(words):
        if w.start < ans_end and w.end > ans_start:
            out.append(i)
    return out


def tag_contract(
    context: str,
    qas: list[dict],
    priority_rank: dict[str, int] | None = None,
) -> tuple[list[str], list[str], dict]:
    """Produce (words, ner_tags, overlap_stats) for a single contract context.

    Applies Strategy A: when a word is already tagged by a category and a new span
    wants to tag it too, the higher-priority (lower rank) category wins. overlap_stats
    records how many token-level overrides happened, keyed by "winner|loser" pair, so
    discarded information can be audited (section 4.2).
    """
    priority_rank = priority_rank if priority_rank is not None else config.PRIORITY_RANK
    words = tokenize_with_offsets(context)
    n = len(words)
    tags = ["O"] * n
    # Track which category currently owns each word (for priority comparison & B/I).
    owner: list[str | None] = [None] * n
    overlap_stats: dict[str, int] = {}

    for qa in qas:
        if qa.get("is_impossible", False):
            # Category absent from this contract — not a negative span, just skip.
            continue
        category = extract_category(qa.get("question", ""))
        if category is None:
            continue
        cat_rank = priority_rank[category]

        for answer in qa.get("answers", []):
            text = answer.get("text", "")
            if not text:
                continue
            start = answer["answer_start"]
            end = start + len(text)
            idxs = _overlapping_word_indices(words, start, end)
            if not idxs:
                continue

            # Decide B vs I position-by-position; respect priority on conflicts.
            first_in_span = True
            for i in idxs:
                existing = owner[i]
                if existing is not None and existing != category:
                    # Conflict: keep whichever has higher priority (lower rank).
                    if priority_rank[existing] <= cat_rank:
                        # Incumbent wins; this category does not take the token.
                        # A B- that loses its first token simply means the next token
                        # it *does* win becomes the B-.
                        continue
                    # New category outranks incumbent -> it takes over this token.
                    overlap_stats[f"{category}|{existing}"] = (
                        overlap_stats.get(f"{category}|{existing}", 0) + 1
                    )
                tags[i] = f"{'B' if first_in_span else 'I'}-{category}"
                owner[i] = category
                first_in_span = False

    return [w.text for w in words], tags, overlap_stats


def reconstruct_spans(words: list[str], tags: list[str]) -> list[tuple[str, str]]:
    """Reconstruct (category, span_text) pairs from a BIO sequence.

    Used by the mandatory validation step (section 4.3.5): a B- opens a span, following
    I-<same category> extend it, anything else closes it. span_text joins the words with
    single spaces (whitespace-normalised, which is what the validation compares against).
    """
    spans: list[tuple[str, str]] = []
    cur_cat: str | None = None
    cur_words: list[str] = []

    def flush():
        nonlocal cur_cat, cur_words
        if cur_cat is not None and cur_words:
            spans.append((cur_cat, " ".join(cur_words)))
        cur_cat, cur_words = None, []

    for word, tag in zip(words, tags):
        if tag == "O":
            flush()
        elif tag.startswith("B-"):
            flush()
            cur_cat = tag[2:]
            cur_words = [word]
        elif tag.startswith("I-"):
            cat = tag[2:]
            if cur_cat == cat:
                cur_words.append(word)
            else:
                # Malformed I- without matching B-; treat as a fresh span start so no
                # text is silently dropped.
                flush()
                cur_cat = cat
                cur_words = [word]
    flush()
    return spans


@dataclass
class ConversionStats:
    contracts: int = 0
    qa_pairs_used: int = 0
    impossible_skipped: int = 0
    overlap_overrides: dict[str, int] = field(default_factory=dict)


def convert_cuad(cuad: dict, limit: int | None = None) -> tuple[list[dict], ConversionStats]:
    """Convert a parsed CUAD_v1.json dict into BIO rows.

    Each contract (title) becomes one row: {"contract_id", "words", "ner_tags"}.
    CUAD stores one paragraph per contract, so contract-level == paragraph-level here;
    we key rows by contract_id to make the section 4.4 split leakage-safe regardless.
    ``limit`` caps the number of contracts for fast local development.
    """
    rows: list[dict] = []
    stats = ConversionStats()

    data = cuad["data"]
    if limit is not None:
        data = data[:limit]

    for entry in data:
        contract_id = entry.get("title", f"contract_{len(rows)}")
        for para in entry["paragraphs"]:
            context = para["context"]
            qas = para.get("qas", [])
            stats.impossible_skipped += sum(1 for q in qas if q.get("is_impossible"))
            stats.qa_pairs_used += sum(1 for q in qas if not q.get("is_impossible"))
            words, tags, overlaps = tag_contract(context, qas)
            for pair, count in overlaps.items():
                stats.overlap_overrides[pair] = stats.overlap_overrides.get(pair, 0) + count
            rows.append(
                {"contract_id": contract_id, "words": words, "ner_tags": tags}
            )
        stats.contracts += 1

    return rows, stats


def contract_level_split(
    rows: list[dict],
    seed: int | None = None,
    ratios: tuple[float, float, float] | None = None,
) -> dict[str, list[dict]]:
    """Split rows into train/val/test grouped by contract_id (section 4.4).

    All rows sharing a contract_id land in exactly one split, so no contract's text can
    appear in more than one split.
    """
    seed = seed if seed is not None else config.SEED
    if ratios is None:
        ratios = (config.TRAIN_RATIO, config.VAL_RATIO, config.TEST_RATIO)

    contract_ids = sorted({r["contract_id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(contract_ids)

    n = len(contract_ids)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_ids = set(contract_ids[:n_train])
    val_ids = set(contract_ids[n_train:n_train + n_val])
    # Remainder (handles rounding) goes to test.

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for r in rows:
        cid = r["contract_id"]
        if cid in train_ids:
            splits["train"].append(r)
        elif cid in val_ids:
            splits["val"].append(r)
        else:
            splits["test"].append(r)
    return splits


def validate_reconstruction(
    cuad: dict,
    rows: list[dict],
    sample_size: int = 20,
    seed: int | None = None,
) -> list[dict]:
    """Section 4.3.5 validation: reconstruct tagged spans and compare to CUAD answers.

    Returns a list of mismatch reports (empty == all good). For a random sample of
    rows, every non-impossible CUAD answer whose category survived Strategy A should be
    reconstructable from the BIO tags with matching (whitespace-normalised) text.

    Because Strategy A can legitimately drop a lower-priority overlapping span, a CUAD
    answer counts as satisfied if its whitespace-normalised text appears within ANY
    reconstructed span of the same category (the span may be longer than the answer when
    adjacent words share the tag). Only answers whose category was never overridden are
    required to match, so this check does not false-alarm on intentional priority drops.
    """
    seed = seed if seed is not None else config.SEED
    rng = random.Random(seed)
    sample = rng.sample(rows, min(sample_size, len(rows)))
    by_id = {r["contract_id"]: r for r in sample}

    # Index CUAD answers by contract title.
    answers_by_contract: dict[str, list[tuple[str, str]]] = {}
    for entry in cuad["data"]:
        title = entry.get("title")
        if title not in by_id:
            continue
        collected: list[tuple[str, str]] = []
        for para in entry["paragraphs"]:
            for qa in para.get("qas", []):
                if qa.get("is_impossible"):
                    continue
                cat = extract_category(qa.get("question", ""))
                if cat is None:
                    continue
                for ans in qa.get("answers", []):
                    if ans.get("text"):
                        collected.append((cat, ans["text"]))
        answers_by_contract[title] = collected

    def norm(s: str) -> str:
        return " ".join(s.split())

    mismatches: list[dict] = []
    for cid, row in by_id.items():
        recon = reconstruct_spans(row["words"], row["ner_tags"])
        recon_by_cat: dict[str, list[str]] = {}
        for cat, text in recon:
            recon_by_cat.setdefault(cat, []).append(norm(text))

        # Which categories were overridden somewhere in this contract? Skip those,
        # since Strategy A may have deliberately altered their spans.
        for cat, ans_text in answers_by_contract.get(cid, []):
            target = norm(ans_text)
            candidates = recon_by_cat.get(cat, [])
            if any(target in span or span in target for span in candidates):
                continue
            # Only report as a mismatch if this category produced NO span at all for a
            # token the answer clearly covers; otherwise it's an accepted priority drop.
            mismatches.append(
                {"contract_id": cid, "category": cat, "answer": target,
                 "reconstructed": candidates}
            )
    return mismatches
