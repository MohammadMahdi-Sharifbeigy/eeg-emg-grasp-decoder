import pytest

from v3.conditions import CONDITIONS, condition_table, describe


def test_names_order_and_eeg_flags():
    assert list(CONDITIONS) == ["causal_cue", "K", "B", "F", "E0", "Eguided"]
    assert [n for n, c in CONDITIONS.items() if c["uses_eeg"]] == ["F", "E0", "Eguided"]
    assert [n for n, c in CONDITIONS.items() if c["trained"]] == ["E0", "Eguided"]


def test_describe_and_table():
    assert describe("B") == "B: K + past task events (primary no-EEG baseline)"
    assert "primary neural candidate" in describe("F")
    t = condition_table()
    assert list(t.condition) == list(CONDITIONS) and {"inputs", "role", "uses_eeg", "source"} <= set(t.columns)
    with pytest.raises(KeyError, match="causal_cue, K, B, F, E0, Eguided"):
        describe("C")
