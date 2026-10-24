import pytest
import torch

import main.dataloader
from v3.teacher import check_teacher_split

RUN = {"train": [1, 3, 5, 6, 7, 8], "val": [9], "test": [2, 4]}


def _ckpt(tmp_path, split=None, where="config"):
    ck = {"best_state": {}, "config": {"model": {}}, "history": {}}
    if split is not None:
        ck[where]["split"] = split
    path = tmp_path / "best.pt"
    torch.save(ck, path)
    return path


def test_same_split_passes_and_is_returned(tmp_path):
    out = check_teacher_split(_ckpt(tmp_path, RUN), 1, RUN)
    assert out["train"] == RUN["train"] and out["source"] == "checkpoint"


def test_teacher_trained_on_a_test_series_raises_naming_it(tmp_path):
    bad = {"train": [1, 2, 5, 6, 7, 8], "val": [9], "test": [3, 4]}
    with pytest.raises(ValueError, match=r"trained on series \[2\]"):
        check_teacher_split(_ckpt(tmp_path, bad, "history"), 1, RUN)


def test_teacher_selected_on_a_test_series_raises(tmp_path):
    bad = {"train": [1, 3, 5, 6, 7, 8], "val": [4], "test": [2, 9]}
    with pytest.raises(ValueError, match=r"selected on series \[4\]"):
        check_teacher_split(_ckpt(tmp_path, bad), 1, RUN)


def test_without_a_stored_split_the_v2_rule_is_used(tmp_path, monkeypatch):
    calls = []

    def fake(p, data_dir=None):
        calls.append(p)
        return {"train": [1, 3, 5, 6, 7, 8], "val": [9], "test": [2, 4]}

    monkeypatch.setattr(main.dataloader, "stratified_split", fake)
    out = check_teacher_split(_ckpt(tmp_path), 5, RUN)
    assert calls == [5] and "stratified_split" in out["source"]
    other = {"train": [1, 2, 3, 5, 6, 7], "val": [8], "test": [4, 9]}
    with pytest.raises(ValueError, match="P5"):
        check_teacher_split(_ckpt(tmp_path), 5, other)
