"""Builds the GOLDEN files: inputs + their exact expected outputs, frozen from the verified text functions.
Re-run ONLY when you deliberately change text processing (then also retrain + update models/manifest.json).
    python tests/fixtures/make_golden.py"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.text import clean_text, format_for_bert

try:
    from src.text import first_350
except ImportError:                       # before first_350 moved into src/text.py: the notebooks' exact definition
    def first_350(t): return " ".join(t.split()[:350])

LONG = " ".join(f"word{i}" for i in range(400))
# (why, input): each one locks in a Phase 1 decision
EDGE_CASES = [
    ("redaction: XXXX", "my account XXXX was closed"),
    ("redaction: full date", "on XX/XX/XXXX they called"),
    ("redaction: date with visible year (year fix)", "on XX/XX/2023 they called"),
    ("redaction: amount", "they charged {$1,234.56} twice"),
    ("redaction: all three together", "On XX/XX/2023 I paid {$50.00} to XXXX"),
    ("word boundary: EXXON is NOT redacted", "I bought gas at EXXON"),
    ("known limitation: glued NumberXXXX stays (0.6% of rows)", "Account NumberXXXX was wrong"),
    ("single X is not a redaction", "plan X was cancelled"),
    ("whitespace: tabs, newlines, repeated spaces", "too\tmany\n\nspaces    here  "),
    ("whitespace only", "     "),
    ("non-string guard: None", None),
    ("non-string guard: NaN-like float", float("nan")),
    ("empty string", ""),
    ("lowercasing", "I WAS CHARGED Twice"),
    ("non-ASCII survives: accent", "café fees were wrong"),
    ("non-ASCII survives: emoji", "terrible service 😡 again"),
    ("first_350: 400 words cut to exactly 350", LONG),
    ("first_350: short text unchanged", "short complaint text"),
    ("format_for_bert: every placeholder", "XXXX on XX/XX/XXXX for {$10.00}"),
    ("format_for_bert: placeholder inside punctuation", "(XXXX), XX/XX/XXXX."),
]


def golden_text():
    cases = []
    for why, raw in EDGE_CASES:
        clean = clean_text(raw)
        cut = first_350(clean)
        cases.append({"why": why, "input": None if raw is None else ("NaN" if raw != raw else raw),
                      "clean_text": clean, "first_350": cut, "format_for_bert": format_for_bert(cut)})
    (HERE / "golden_text.json").write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(cases)


def bert_golden(n_longest=10, n_total=50):
    """50 of the 1,000 Day 13 Colab rows (raw text + KerasHub token IDs): the 10 longest + 40 evenly spaced."""
    raw = json.loads((ROOT / "models" / "ref_raw_texts.json").read_text(encoding="utf-8"))
    ids = np.load(ROOT / "models" / "ref_token_ids.npy")
    lengths = (ids != 0).sum(axis=1)
    longest = list(np.argsort(-lengths, kind="stable")[:n_longest])
    rest = [i for i in np.linspace(0, len(raw) - 1, n_total * 2, dtype=int) if i not in longest]
    chosen = sorted(set(longest) | set(rest[: n_total - n_longest]))
    rows = [{"row": int(i), "raw_text": raw[i], "keras_hub_ids": ids[i][ids[i] != 0].tolist()} for i in chosen]
    (HERE / "bert_token_golden.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    shutil.copy(ROOT / "models" / "bert_vocab.txt", HERE / "bert_vocab.txt")
    return len(rows), int(lengths[longest[0]])


if __name__ == "__main__":
    print("golden_text cases:", golden_text())
    print("bert rows, longest (tokens):", bert_golden())
