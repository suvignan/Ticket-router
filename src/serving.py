"""What the API runs. Tests call THESE functions, never a test-only copy. No pandas here (see test_serving.py)."""
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
from tokenizers import BertWordPieceTokenizer

from src.constants import ROOT
from src.text import clean_text, first_350, format_for_bert
from src.utils import verify_checksums

BERT_VOCAB = ROOT / "models" / "bert_vocab.txt"
MANIFEST = ROOT / "models" / "manifest.json"
MAX_LEN = 512


@lru_cache(maxsize=None)
def bert_tokenizer(vocab_path: Path = BERT_VOCAB) -> BertWordPieceTokenizer:
    """Loaded once per process. Same WordPiece as KerasHub (100% token-ID parity, Day 13), without TensorFlow."""
    tok = BertWordPieceTokenizer(str(vocab_path), lowercase=True)     # adds [CLS] ... [SEP]
    tok.enable_truncation(MAX_LEN)
    return tok


def encode_for_bert(raw_text, vocab_path: Path = BERT_VOCAB) -> tuple[np.ndarray, np.ndarray]:
    """raw complaint -> (token_ids, padding_mask), shape (1, n), real length (no pad-to-512: same argmax, faster)."""
    ids = np.array([bert_tokenizer(vocab_path).encode(format_for_bert(first_350(clean_text(raw_text)))).ids], dtype=np.int32)
    return ids, (ids != 0).astype(np.int32)


def _labels(files) -> list[str]:
    return json.loads(files["label_order"].read_text(encoding="utf-8"))


class LRBundle:
    """TF-IDF + LogisticRegression (logreg_b). Reads the FULL text."""
    MAIN = "logreg"                                             # its sha256 goes in every response

    def __init__(self, files):
        self.vec = joblib.load(files["tfidf"])
        self.model = joblib.load(files["logreg"])
        self.labels = _labels(files)
        if list(self.model.classes_) != self.labels:            # the Day 9 bug: probabilities under the wrong names
            raise RuntimeError(f"LR classes_ {list(self.model.classes_)} != label_order {self.labels}")

    def predict_proba(self, raw_text) -> np.ndarray:
        return self.model.predict_proba(self.vec.transform([clean_text(raw_text)]))[0]


class BertBundle:
    """Fine-tuned BERT exported to ONNX. Reads the first 350 words."""
    MAIN = "onnx"

    def __init__(self, files):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1                             # serving setting (Day 13): 1 request = 1 core, no oversubscription
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(files["onnx"]), so, providers=["CPUExecutionProvider"])
        self.vocab = files["vocab"]
        self.labels = _labels(files)

    def predict_proba(self, raw_text) -> np.ndarray:
        ids, mask = encode_for_bert(raw_text, self.vocab)
        logits = self.sess.run(None, {"token_ids": ids, "segment_ids": np.zeros_like(ids), "padding_mask": mask})[0]
        if logits.shape != (1, len(self.labels)):
            raise RuntimeError(f"BERT output shape {logits.shape}, expected (1, {len(self.labels)})")
        z = logits[0].astype(np.float64)
        e = np.exp(z - z.max())
        return e / e.sum()


BUNDLES = {"lr": LRBundle, "bert": BertBundle}


def load_bundle(cfg: dict, manifest: Path = MANIFEST, root: Path = ROOT):
    """cfg = params.yaml `serving`. Verifies EVERY file against the manifest BEFORE loading anything:
    on a mismatch it raises and the service refuses to start. Better no service than the wrong model.
    Tests pass their own manifest + root (a fixture-trained model is the test's own champion)."""
    files = cfg["files"][cfg["model"]]
    checksums = verify_checksums(files.values(), manifest, root)
    cls = BUNDLES[cfg["model"]]
    bundle = cls({key: root / path for key, path in files.items()})
    bundle.name, bundle.checksums, bundle.model_sha256 = cfg["model"], checksums, checksums[files[cls.MAIN]]
    p = bundle.predict_proba("my credit card was charged twice")    # warm-up + shape check: fail now, not on request 1
    if p.shape != (len(bundle.labels),) or abs(p.sum() - 1) > 1e-4:
        raise RuntimeError(f"{cfg['model']} warm-up gave shape {p.shape}, sum {p.sum()}")
    return bundle
