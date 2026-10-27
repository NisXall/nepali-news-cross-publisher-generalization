import pandas as pd

from src.near_dup_audit import audit, shingles


def test_shingles_use_fixed_word_window():
    assert shingles("a b c d e f") == {"a b c d e", "b c d e f"}


def test_exact_jaccard_threshold_is_verified():
    base = " ".join(f"w{i}" for i in range(30))
    near = " ".join(f"x{i}" if i == 29 else f"w{i}" for i in range(30))
    far = " ".join(f"z{i}" for i in range(30))
    frame = pd.DataFrame({
        "id": ["base", "near", "far"],
        "text": [base, near, far],
        "label": ["A", "A", "B"],
        "source": ["one", "two", "three"],
    })
    result = audit(frame, "text", "id", "label", "source")
    assert result["qualifying_pairs"] == 1
    assert result["groups"][0]["labels"] == ["A"]
