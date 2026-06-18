from __future__ import annotations

import re
from collections.abc import Iterable


EMPTY_TACTIC = "<EMPTY_TACTIC>"
UNKNOWN_TACTIC = "<UNK_TACTIC>"
TACTIC_TOKEN_RE = re.compile(r"[A-Za-z0-9_.'!?]+")


def normalize_tactic(raw: str) -> str:
    text = raw.strip()
    if not text:
        return EMPTY_TACTIC

    match = TACTIC_TOKEN_RE.search(text)
    if match is None:
        return EMPTY_TACTIC
    return match.group(0)


def build_tactic_vocab(labels: Iterable[str]) -> dict[str, int]:
    vocab = {UNKNOWN_TACTIC: 0}
    for index, label in enumerate(sorted(set(labels)), start=1):
        vocab[label] = index
    return vocab


def label_example(raw_tactic: str) -> dict[str, object]:
    tactic_name = normalize_tactic(raw_tactic)
    return {
        "tactic_raw": raw_tactic,
        "tactic_name": tactic_name,
    }


def encode_tactic_name(tactic_name: str, tactic_vocab: dict[str, int]) -> int:
    return tactic_vocab.get(tactic_name, tactic_vocab[UNKNOWN_TACTIC])


# ---------------------------------------------------------------------------
# Tactic arity registry (strict static dictionary, no data-inference fallback)
# ---------------------------------------------------------------------------

TACTIC_ARITY: dict[str, int] = {
    "simp": 0,
    "ring": 0,
    "norm_num": 0,
    "omega": 0,
    "decide": 0,
    "trivial": 0,
    "contradiction": 0,
    "linarith": 0,
    "nlinarith": 0,
    "tauto": 0,
    "aesop": 0,
    "aesop?": 0,
    "apply": 1,
    "exact": 1,
    "rw": 1,
    "rewrite": 1,
    "have": 2,
    "calc": 0,
    "intro": 1,
    "intros": 0,
    "ext": 1,
    "cases": 1,
    "induction": 1,
    "constructor": 0,
    "use": 1,
    "refine": 1,
    "specialize": 1,
    "obtain": 1,
    "simp_all": 0,
    "norm_cast": 0,
    "push_cast": 0,
    "ring_nf": 0,
    "field_simp": 0,
    "positivity": 0,
    "gcongr": 0,
    "congr": 0,
    "funext": 0,
    "rfl": 0,
    "assumption": 0,
    "left": 0,
    "right": 0,
    "exfalso": 0,
    "by_contra": 0,
    "push_neg": 0,
    "contrapose": 0,
    "absurd": 1,
    "replace": 1,
    "conv": 0,
    "change": 1,
    "show": 1,
    "suffices": 1,
    "let": 2,
    "set": 1,
    "rcases": 1,
    "rintro": 0,
    "simp?": 0,
    "exact?": 0,
    "apply?": 0,
}

DEFAULT_ARITY: int = 1


def get_tactic_arity(tactic_name: str) -> int:
    """Return the expected number of pointer-selected arguments for *tactic_name*."""
    return TACTIC_ARITY.get(tactic_name, DEFAULT_ARITY)


# ---------------------------------------------------------------------------
# Best-effort tactic argument extraction
# ---------------------------------------------------------------------------

_ARG_TOKEN_RE = re.compile(r"[A-Za-z0-9_.']+")
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")
_NAMED_ARGUMENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_']*\s*:=")
_TACTIC_KEYWORDS = {
    "at",
    "by",
    "generalizing",
    "in",
    "only",
    "using",
    "with",
}
_ARGUMENT_TARGET_MARKERS = (" at ", " using ", " with ", " by ", " generalizing ")
_REWRITE_LIKE_TACTICS = {
    "rw",
    "rewrite",
    "erw",
    "nth_rewrite",
    "nth_rw",
    "rw_mod_cast",
    "rwa",
    "simp",
    "simp!",
    "simp?",
    "simp_all",
    "simp_all!",
    "simp_rw",
    "simpa",
    "simpa!",
    "simpa?",
}
_UNARY_PROOF_TACTICS = {
    "absurd",
    "apply",
    "apply_fun",
    "apply_mod_cast",
    "change",
    "convert",
    "convert_to",
    "exact",
    "exact_mod_cast",
    "fapply",
    "refine",
    "refine'",
    "show",
}
_BINDER_OR_STRUCTURAL_TACTICS = {
    "all_goals",
    "any_goals",
    "case",
    "case'",
    "classical",
    "constructor",
    "ext",
    "ext1",
    "funext",
    "intro",
    "intros",
    "introv",
    "left",
    "next",
    "rfl",
    "right",
    "rintro",
    "trivial",
}


def _strip_markup(raw: str) -> str:
    return _HTML_TAG_RE.sub("", raw)


def _argument_remainder(raw: str, tactic_name: str) -> str:
    match = TACTIC_TOKEN_RE.search(raw.strip())
    if match is None or match.group(0) != tactic_name:
        return ""
    return raw.strip()[match.end():].strip()


def _strip_target_clause(text: str) -> str:
    padded = f" {text} "
    positions = [
        padded.find(marker)
        for marker in _ARGUMENT_TARGET_MARKERS
        if padded.find(marker) != -1
    ]
    if not positions:
        return text.strip()
    return padded[: min(positions)].strip()


def _identifier_tokens(text: str) -> list[str]:
    text = _NAMED_ARGUMENT_RE.sub(" ", text)
    tokens: list[str] = []
    for token_match in _ARG_TOKEN_RE.finditer(text):
        token = token_match.group(0).strip("'.")
        if not token or token in _TACTIC_KEYWORDS:
            continue
        if token == "_" or token.startswith("_") or token.startswith("?"):
            continue
        if token.isdigit():
            continue
        tokens.append(token)
    return tokens


def _split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    pairs = {"[": "]", "(": ")", "⟨": "⟩", "{": "}"}
    closing = set(pairs.values())

    for char in text:
        if char in pairs:
            depth += 1
        elif char in closing and depth > 0:
            depth -= 1
        if char == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(char)

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _bracketed_argument_tokens(text: str) -> list[str]:
    args: list[str] = []
    for open_ch, close_ch in zip("[⟨(", "]⟩)"):
        start = text.find(open_ch)
        if start == -1:
            continue
        end = text.rfind(close_ch)
        if end > start:
            content = text[start + 1 : end]
        else:
            content = text[start + 1 :]
        for item in _split_top_level_commas(content):
            item_tokens = _identifier_tokens(item)
            if item_tokens:
                args.append(item_tokens[0])
        break
    return args


def _first_identifier_token(text: str) -> list[str]:
    tokens = _identifier_tokens(text)
    return tokens[:1]


def parse_tactic_arguments(raw: str) -> tuple[str, list[str]]:
    """Extract the tactic family name and a list of argument tokens.

    Examples
    --------
    >>> parse_tactic_arguments("rw [foo, bar]")
    ('rw', ['foo', 'bar'])
    >>> parse_tactic_arguments("apply h1")
    ('apply', ['h1'])
    >>> parse_tactic_arguments("simp only [h1, h2]")
    ('simp', ['h1', 'h2'])
    >>> parse_tactic_arguments("simp")
    ('simp', [])
    """
    text = _strip_markup(raw).strip()
    if not text:
        return EMPTY_TACTIC, []

    tactic_match = TACTIC_TOKEN_RE.search(text)
    if tactic_match is None:
        return EMPTY_TACTIC, []

    tactic_name = tactic_match.group(0)
    remainder = _argument_remainder(text, tactic_name)

    if not remainder:
        return tactic_name, []

    if tactic_name in _BINDER_OR_STRUCTURAL_TACTICS:
        return tactic_name, []

    if tactic_name in _REWRITE_LIKE_TACTICS:
        bracketed_args = _bracketed_argument_tokens(remainder)
        if bracketed_args:
            return tactic_name, bracketed_args
        return tactic_name, _identifier_tokens(_strip_target_clause(remainder))

    if tactic_name in {"cases", "rcases", "induction"}:
        return tactic_name, _first_identifier_token(_strip_target_clause(remainder))

    if tactic_name in _UNARY_PROOF_TACTICS:
        return tactic_name, _first_identifier_token(remainder)

    bracketed_args = _bracketed_argument_tokens(remainder)
    if bracketed_args:
        return tactic_name, bracketed_args

    return tactic_name, _identifier_tokens(remainder)
