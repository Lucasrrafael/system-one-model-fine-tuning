import pytest

from src.core.targets import target_vector


def test_score_index_and_fraction():
    q = {"type": "score", "criteria": ["low", "medium", "high"]}
    assert target_vector(q, ["0", "1", "2"], expected=2) == [0.0, 0.0, 1.0]
    assert target_vector(q, ["0", "1", "2"], expected=0.5) == [0.5, 0.5, 0.0]
    with pytest.raises(ValueError):
        target_vector(q, ["0", "1", "2"], expected=3)


def test_target_follows_option_ids_not_position():
    choice = {"type": "choice", "criteria": {"zeta": "z", "alpha": "a"}}
    assert target_vector(choice, ["alpha", "zeta"], expected="zeta") == [0.0, 1.0]
    noul = {"type": "noul"}
    assert target_vector(noul, ["true", "false"], expected=False) == [0.0, 1.0]
    assert target_vector(noul, ["false", "true"], expected="true") == [0.0, 1.0]


def test_gold_and_smoothing():
    q = {"type": "choice", "criteria": {"a": "", "b": ""}}
    assert target_vector(q, ["a", "b"], gold={"b": 3, "a": 1}) == [0.25, 0.75]
    assert target_vector(q, ["a", "b"], expected="a", label_smoothing=0.2) == pytest.approx([0.9, 0.1])


def test_clef_option_order_and_canonical_mapping(clef_files):
    """A ordem real do encoder do clef: choice alfabético, noul true/false, score 0..K-1."""
    from transformers import AutoTokenizer

    from src.families import clef

    jsm = clef._jsm(clef_files)
    tok = AutoTokenizer.from_pretrained(str(clef_files))
    questions = {
        "team": {"type": "choice", "instructions": "Which team?", "criteria": {"zeta": "Z team", "alpha": "A team"}},
        "down": {"type": "noul", "instructions": "Is it down?"},
        "priority": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]},
    }
    enc = jsm.encode_record(tok, {"state": "Checkout is failing.", "questions": questions})
    ids = {q.question_id: list(q.option_ids) for q in enc.questions}
    assert ids == {"team": ["alpha", "zeta"], "down": ["true", "false"], "priority": ["0", "1", "2"]}
    assert clef._canonical_order(questions["team"], ids["team"]) == [1, 0]
    assert clef._canonical_order(questions["down"], ids["down"]) == [1, 0]
    assert clef._canonical_order(questions["priority"], ids["priority"]) == [0, 1, 2]
