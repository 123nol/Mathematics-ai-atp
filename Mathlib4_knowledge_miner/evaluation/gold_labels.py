"""
evaluation/gold_labels.py
=========================

Gold-label extraction and argument classification for the LeanDojo benchmark.

CRITICAL: LeanDojo tactic strings use HTML anchor markup to annotate library
lemma references:

    rw [<a>Nat.add_comm</a>, h]
    apply <a>Finset.sum_le_sum</a>
    simp [<a>WithLp.prod_norm_eq_add</a> (hp.symm)]

Gold library lemma names are ONLY those inside <a>...</a> tags. All other
tokens (local hypotheses, numeric literals, sub-expressions like `hp.symm`)
are IGNORED for gold-label purposes.


This module is self-contained and has no dependency on torch_geometric.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Primary gold-label extraction: <a> anchor tags in tactic strings
# ---------------------------------------------------------------------------

# Matches <a ...>lemma.Name</a>  (the LeanDojo hyperlink format)
_ANCHOR_RE = re.compile(r"<a[^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)

# Tactic name token (used for tactic family identification only)
_TACTIC_TOKEN_RE = re.compile(r"[A-Za-z0-9_.'!?]+")
_ARG_TOKEN_RE = re.compile(r"[A-Za-z0-9_.']+")
_OPERATOR_RE = re.compile(
    r"[+\-*/=<>!@#$%^&|~]|λ|→|←|↔|∀|∃|⟩|⟨"
)


class ArgCategory:
    LOCAL_HYPOTHESIS = "local_hypothesis"
    LIBRARY_LEMMA = "library_lemma"
    RAW_EXPRESSION = "raw_expression"
    UNRESOLVED = "unresolved"
    GRAPH_NON_CANDIDATE = "graph_non_candidate"


@dataclass
class ClassifiedArg:
    token: str
    category: str
    lemma_id: int  # -1 if not a library lemma


@dataclass
class TacticGoldLabel:
    """Gold label for one tactic in the validation set."""

    tactic_raw: str
    tactic_name: str
    args: list[ClassifiedArg]
    # Deduplicated list of library lemma IDs from <a> tags (may be empty)
    library_lemma_ids: list[int]

    @property
    def has_library_lemma(self) -> bool:
        return bool(self.library_lemma_ids)

    @property
    def is_local_only(self) -> bool:
        """True only when the tactic has NO <a> tags at all and all resolved
        args are local hypotheses.  Used only for exclusion-reason labelling.
        """
        resolved = [a for a in self.args if a.category != ArgCategory.UNRESOLVED]
        if not resolved:
            return False
        return all(a.category == ArgCategory.LOCAL_HYPOTHESIS for a in resolved)


# ---------------------------------------------------------------------------
# Tactic name parsing
# ---------------------------------------------------------------------------

_EMPTY_TACTIC = "<EMPTY_TACTIC>"


def parse_tactic_arguments(raw: str) -> tuple[str, list[str]]:
    """Extract the tactic family name and argument token list.

    Note: This strips <a>...</a> markup before token extraction so that
    the `a` characters from the HTML tags do not appear as argument tokens.
    The returned argument list is used for display/classification only —
    gold lemma IDs are separately extracted by extract_gold_labels via
    _extract_anchor_names().
    """
    text = raw.strip()
    if not text:
        return _EMPTY_TACTIC, []

    # Strip anchor tags so that `a` from <a>/</a> doesn't appear as a token
    clean = _ANCHOR_RE.sub(lambda m: m.group(1), text)

    tactic_match = _TACTIC_TOKEN_RE.search(clean)
    if tactic_match is None:
        return _EMPTY_TACTIC, []

    tactic_name = tactic_match.group(0)
    remainder = clean[tactic_match.end():].strip()

    for keyword in ("only", "with", "using", "at"):
        if remainder.startswith(keyword):
            remainder = remainder[len(keyword):].strip()

    args: list[str] = []
    bracket_content: str | None = None
    for open_ch, close_ch in zip("[⟨(", "]⟩)"):
        start = remainder.find(open_ch)
        if start != -1:
            end = remainder.rfind(close_ch)
            if end > start:
                bracket_content = remainder[start + 1: end]
            else:
                bracket_content = remainder[start + 1:]
            break

    if bracket_content is not None:
        for token_match in _ARG_TOKEN_RE.finditer(bracket_content):
            args.append(token_match.group(0))
    elif remainder:
        for token_match in _ARG_TOKEN_RE.finditer(remainder):
            args.append(token_match.group(0))

    return tactic_name, args


def _extract_anchor_names(tactic_raw: str) -> list[str]:
    """Return all names wrapped in <a>...</a> tags in tactic_raw.

    These are the ONLY gold library lemma candidates for LeanDojo data.
    """
    return [m.strip() for m in _ANCHOR_RE.findall(tactic_raw)]


# ---------------------------------------------------------------------------
# Argument classification (used for fallback / display, not for gold IDs)
# ---------------------------------------------------------------------------

def _looks_like_raw_expression(token: str) -> bool:
    if token.lstrip("+-").isdigit():
        return True
    if token and token[0].isdigit():
        return True
    if _OPERATOR_RE.search(token):
        return True
    return False


def classify_argument(
    token: str,
    *,
    hypothesis_names: set[str],
    lemma_name_index: dict[str, int],
) -> ClassifiedArg:
    """Classify a single tactic argument token (used for display only)."""
    if _looks_like_raw_expression(token):
        return ClassifiedArg(token=token, category=ArgCategory.RAW_EXPRESSION, lemma_id=-1)
    if token in hypothesis_names:
        return ClassifiedArg(token=token, category=ArgCategory.LOCAL_HYPOTHESIS, lemma_id=-1)
    lemma_id = lemma_name_index.get(token, -1)
    if lemma_id >= 0:
        return ClassifiedArg(token=token, category=ArgCategory.LIBRARY_LEMMA, lemma_id=lemma_id)
    return ClassifiedArg(token=token, category=ArgCategory.UNRESOLVED, lemma_id=-1)


# ---------------------------------------------------------------------------
# Primary API
# ---------------------------------------------------------------------------

def extract_gold_labels(
    tactic_raw: str,
    proof_state: str,
    *,
    lemma_name_index: dict[str, int],
) -> TacticGoldLabel:
    """Extract gold library lemma IDs for one LeanDojo proof step.
    """
    tactic_name, tokens = parse_tactic_arguments(tactic_raw)
    hypothesis_names = _extract_hypothesis_names(proof_state)

    # Classify tokens for display/debug (does NOT set gold label IDs)
    classified: list[ClassifiedArg] = [
        classify_argument(
            token,
            hypothesis_names=hypothesis_names,
            lemma_name_index=lemma_name_index,
        )
        for token in tokens
    ]

    # ---------------------------------------------------------------
    # PRIMARY: use <a> anchor names as gold labels
    # ---------------------------------------------------------------
    anchor_names = _extract_anchor_names(tactic_raw)

    seen: set[int] = set()
    library_ids: list[int] = []

    for name in anchor_names:
        lid = lemma_name_index.get(name, -1)
        if lid >= 0 and lid not in seen:
            seen.add(lid)
            library_ids.append(lid)

    # ---------------------------------------------------------------
    # FALLBACK (for tests that use plain tactics without <a> tags):
    # if no anchors present at all, fall back to token classification.
    # This preserves backward compatibility with unit tests.
    # ---------------------------------------------------------------
    if not anchor_names:
        for arg in classified:
            if arg.category == ArgCategory.LIBRARY_LEMMA and arg.lemma_id >= 0:
                if arg.lemma_id not in seen:
                    seen.add(arg.lemma_id)
                    library_ids.append(arg.lemma_id)

    return TacticGoldLabel(
        tactic_raw=tactic_raw,
        tactic_name=tactic_name,
        args=classified,
        library_lemma_ids=library_ids,
    )


def _extract_hypothesis_names(proof_state: str) -> set[str]:
    """Extract hypothesis names from the proof state string."""
    names: set[str] = set()
    turnstile = None
    for ts in ("⊢", "|-"):
        if ts in proof_state:
            turnstile = ts
            break

    if turnstile is None:
        return names

    hyp_block = proof_state.split(turnstile, 1)[0].strip()
    for line in hyp_block.splitlines():
        line = line.strip()
        if not line:
            continue
        if " : " in line:
            name = line.split(" : ")[0].strip()
        elif ":" in line:
            name = line.split(":")[0].strip()
        else:
            name = line
        if name:
            names.add(name)

    return names


# ---------------------------------------------------------------------------
# Category counting helpers
# ---------------------------------------------------------------------------

@dataclass
class ExclusionSummary:
    total_processed: int
    local_hypothesis_only: int
    no_library_lemma_target: int
    unresolved_only: int
    raw_expression_only: int
    has_library_lemma: int

    def __str__(self) -> str:
        lines = [
            f"  Total proof states processed  : {self.total_processed}",
            f"  → Has library lemma target    : {self.has_library_lemma}  (denominator)",
            f"  → Excluded — local hyp only   : {self.local_hypothesis_only}",
            f"  → Excluded — unresolved only  : {self.unresolved_only}",
            f"  → Excluded — raw expr only    : {self.raw_expression_only}",
            f"  → Excluded — no lib. lemma    : {self.no_library_lemma_target}",
        ]
        return "\n".join(lines)
