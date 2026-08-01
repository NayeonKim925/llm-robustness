"""Unit tests for robust label parsing (run: python -m pytest -q)."""

from llm_robustness import labels


def test_exact_labels():
    for name in labels.LABELS:
        assert labels.parse_label(name) == name


def test_case_and_whitespace():
    assert labels.parse_label("  DENY\n") == "deny"
    assert labels.parse_label("Comment.") == "comment"


def test_label_inside_sentence():
    assert labels.parse_label("The stance is clearly support here.") == "support"


def test_word_boundary_not_substring():
    # "supportive" must NOT be read as "support"; no other label word present.
    assert labels.parse_label("supportive") == labels.INVALID
    # "commentary" must NOT be read as "comment".
    assert labels.parse_label("commentary") == labels.INVALID


def test_deterministic_priority_order():
    # If several label words appear, the canonical order wins, every time.
    out = {labels.parse_label("support deny query comment") for _ in range(50)}
    assert out == {"support"}


def test_invalid_on_empty_or_unrelated():
    assert labels.parse_label("") == labels.INVALID
    assert labels.parse_label("banana") == labels.INVALID


def test_id_roundtrip():
    for i, name in enumerate(labels.LABELS):
        assert labels.to_id(name) == i
        assert labels.ID_TO_LABEL[i] == name
