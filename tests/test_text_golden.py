"""GOLDEN tests: text processing must reproduce the frozen outputs EXACTLY (train/serve skew hides here).
A deliberate change = regenerate with tests/fixtures/make_golden.py + retrain + update models/manifest.json."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(ROOT))
from src.serving import encode_for_bert
from src.text import clean_text, first_350, format_for_bert, is_no_content
from src.utils import read_yaml, sha256_file

CASES = json.loads((FIXTURES / "golden_text.json").read_text(encoding="utf-8"))
BERT_ROWS = json.loads((FIXTURES / "bert_token_golden.json").read_text(encoding="utf-8"))
VOCAB = FIXTURES / "bert_vocab.txt"
MANIFEST_VOCAB_SHA = "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"   # models/manifest.json


@pytest.mark.parametrize("case", CASES, ids=[c["why"] for c in CASES])
def test_text_golden(case):
    raw = float("nan") if case["input"] == "NaN" else case["input"]
    assert clean_text(raw) == case["clean_text"]
    assert first_350(case["clean_text"]) == case["first_350"]
    assert format_for_bert(case["first_350"]) == case["format_for_bert"]


def test_first_350_case_really_truncates():
    long = next(c for c in CASES if c["why"].startswith("first_350: 400"))
    assert len(long["clean_text"].split()) == 400 and len(long["first_350"].split()) == 350   # guard: the golden tests a real cut


def test_vocab_is_the_deployed_one():
    assert sha256_file(VOCAB) == MANIFEST_VOCAB_SHA      # the test's vocabulary can never drift from the deployed one


def test_bert_tokens_match_keras_hub():
    assert len(BERT_ROWS) == 50 and max(len(r["keras_hub_ids"]) for r in BERT_ROWS) == 504   # guard: includes the longest
    wrong = []
    for r in BERT_ROWS:
        ids, mask = encode_for_bert(r["raw_text"], vocab_path=VOCAB)
        if ids[0].tolist() != r["keras_hub_ids"] or mask.sum() != ids.shape[1]:
            wrong.append(r["row"])
    assert not wrong, f"token IDs differ from KerasHub on reference rows {wrong}"


# is_no_content: ONE function for ingestion (drops the row) and the API (review, model not called)
JUNK_RULES = read_yaml(ROOT / "params.yaml")["junk_rules"]
NO_CONTENT = [                                    # (raw text, is junk, why)
    (float("nan"), True, "NaN narrative"),
    (None, True, "None"),
    ("", True, "empty"),
    ("   \n\t ", True, "whitespace only"),
    ("XXXX", True, "redaction only"),
    ("XX/XX/XXXX XXXX", True, "redacted date + name only"),
    ("See attached.", True, "short attachment note"),
    ("  PLEASE SEE ATTACHMENT  ", True, "case-insensitive, stripped"),
    ("see attached document regarding my mortgage", False, "attachment phrase but long enough (> 30)"),
    ("XXXX charged me twice", False, "redaction + real words"),
    ("paid debt", False, "shortest real narrative: content (too_short is a different rule)"),
    ("x", False, "lowercase x is a letter, not a redaction"),
]


@pytest.mark.parametrize("raw, junk, why", NO_CONTENT, ids=[c[2] for c in NO_CONTENT])
def test_is_no_content(raw, junk, why):
    assert is_no_content(raw, JUNK_RULES) is junk
