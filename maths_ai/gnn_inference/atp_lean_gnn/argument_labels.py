from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .graph import DAGBuilder
from .labels import get_tactic_arity, parse_tactic_arguments
from .pyg import build_premise_mask


LOCAL_HYPOTHESIS = "local_hypothesis"
LIBRARY_LEMMA = "library_lemma"
GRAPH_NON_CANDIDATE = "graph_non_candidate"
UNRESOLVED = "unresolved"
RAW_EXPRESSION = "raw_expression"
NO_ARGUMENT = "no_argument"

_RAW_EXPRESSION_MARKERS = (
    " = ",
    " + ",
    " - ",
    " * ",
    " / ",
    " -> ",
    " → ",
    " ↔ ",
    " ∧ ",
    " ∨ ",
    " ≤ ",
    " ≥ ",
    " < ",
    " > ",
    ":=",
    " fun ",
    " λ",
    "∀",
)
_BRACKETED_RE = re.compile(r"[\[⟨(].*[\]⟩)]")


@dataclass(frozen=True)
class ArgumentResolution:
    token: str
    category: str
    node_id: int = -1
    lemma_id: int = -1


@dataclass(frozen=True)
class ArgumentLabelAnalysis:
    tactic_name: str
    arg_tokens: list[str]
    resolutions: list[ArgumentResolution]
    has_raw_expression_argument: bool


def _label_to_ids(dag: DAGBuilder) -> dict[str, list[int]]:
    label_to_ids: dict[str, list[int]] = {}
    for node in dag.nodes:
        label_to_ids.setdefault(node.label, []).append(node.id)
    return label_to_ids


def resolve_argument_tokens(
    dag: DAGBuilder,
    arg_tokens: Iterable[str],
    *,
    lemma_name_index: dict[str, int] | None = None,
    premise_mask: list[bool] | None = None,
) -> list[ArgumentResolution]:
    """Resolve tactic argument tokens into clean local/lemma categories.

    Local graph nodes are accepted only when they are allowed by
    ``premise_mask``.  A token that appears only in the goal-side graph is
    reported as ``graph_non_candidate`` rather than being used as a training
    target.
    """
    if premise_mask is None:
        premise_mask = build_premise_mask(dag)

    label_to_ids = _label_to_ids(dag)
    resolutions: list[ArgumentResolution] = []
    for token in arg_tokens:
        local_ids = [
            node_id
            for node_id in label_to_ids.get(token, [])
            if 0 <= node_id < len(premise_mask) and premise_mask[node_id]
        ]
        if local_ids:
            resolutions.append(
                ArgumentResolution(
                    token=token,
                    category=LOCAL_HYPOTHESIS,
                    node_id=local_ids[0],
                )
            )
            continue

        if lemma_name_index is not None and token in lemma_name_index:
            resolutions.append(
                ArgumentResolution(
                    token=token,
                    category=LIBRARY_LEMMA,
                    lemma_id=lemma_name_index[token],
                )
            )
            continue

        if token in label_to_ids:
            resolutions.append(
                ArgumentResolution(
                    token=token,
                    category=GRAPH_NON_CANDIDATE,
                    node_id=label_to_ids[token][0],
                )
            )
            continue

        resolutions.append(ArgumentResolution(token=token, category=UNRESOLVED))

    return resolutions


def looks_like_raw_expression_argument(raw_tactic: str, tactic_name: str, arg_tokens: list[str]) -> bool:
    if not arg_tokens:
        return False

    expected_arity = get_tactic_arity(tactic_name)
    if expected_arity == 0:
        return False

    remainder = raw_tactic.strip()
    tactic_pos = remainder.find(tactic_name)
    if tactic_pos != -1:
        remainder = remainder[tactic_pos + len(tactic_name) :].strip()

    if tactic_name in {"rw", "rewrite", "simp", "simp_all", "simp?"} and _BRACKETED_RE.search(remainder):
        return False

    padded = f" {remainder} "
    if any(marker in padded for marker in _RAW_EXPRESSION_MARKERS):
        return True

    return expected_arity == 1 and len(arg_tokens) > 1 and tactic_name in {
        "apply",
        "exact",
        "refine",
        "change",
        "show",
    }


def analyze_argument_labels(
    *,
    raw_tactic: str,
    dag: DAGBuilder,
    lemma_name_index: dict[str, int] | None = None,
    premise_mask: list[bool] | None = None,
) -> ArgumentLabelAnalysis:
    tactic_name, arg_tokens = parse_tactic_arguments(raw_tactic)
    resolutions = resolve_argument_tokens(
        dag,
        arg_tokens,
        lemma_name_index=lemma_name_index,
        premise_mask=premise_mask,
    )
    return ArgumentLabelAnalysis(
        tactic_name=tactic_name,
        arg_tokens=arg_tokens,
        resolutions=resolutions,
        has_raw_expression_argument=looks_like_raw_expression_argument(
            raw_tactic,
            tactic_name,
            arg_tokens,
        ),
    )
