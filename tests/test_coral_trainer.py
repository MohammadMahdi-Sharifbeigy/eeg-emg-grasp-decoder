import os
import sys
import tempfile
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath('.'))

from main import (
    CORALNet,
    CORALLoss,
    CORALTrainer,
    CORALTrainConfig,
    build_coral_loss_from_config,
)

def test_coral_trainer_smoke_and_resume():
    torch.manual_seed(42)
    B, T, n_eeg, n_muscles, kin_dim = 4, 50, 32, 5, 10
    eeg = torch.randn(B * 3, T, n_eeg)
    kin = torch.randn(B * 3, T, kin_dim)
    emg = torch.randn(B * 3, T, n_muscles)

    ds = TensorDataset(eeg, kin, emg)
    train_loader = DataLoader(ds, batch_size=4)
    val_loader = DataLoader(ds, batch_size=4)

    model = CORALNet(
        n_eeg_channels=n_eeg,
        n_muscles=n_muscles,
        n_synergies=3,
        d_model=32,
        n_layers=1,
        d_state=8,
        use_kinematics=True,
        kin_dim=kin_dim,
    )
    criterion = CORALLoss(w_ccc=1.0, w_pearson=0.5, w_diff=0.1, w_syn=0.1)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg1 = CORALTrainConfig(
            epochs=2,
            lr=1e-3,
            checkpoint_dir=tmpdir,
            checkpoint_every=1,
            use_amp=False,
            device="cpu",
            resume=False,
        )
        trainer1 = CORALTrainer(model, criterion, config=cfg1)
        res1 = trainer1.train(train_loader, val_loader, verbose=False)

        assert os.path.exists(os.path.join(tmpdir, "last.pt"))
        assert os.path.exists(os.path.join(tmpdir, "best.pt"))
        assert len(res1.history["epoch"]) == 2

        # Resume to epoch 4
        cfg2 = CORALTrainConfig(
            epochs=4,
            lr=1e-3,
            checkpoint_dir=tmpdir,
            checkpoint_every=1,
            use_amp=False,
            device="cpu",
            resume=True,
        )
        trainer2 = CORALTrainer(model, criterion, config=cfg2)
        res2 = trainer2.train(train_loader, val_loader, verbose=False)

        assert res2.resumed_from_epoch == 2
        assert len(res2.history["epoch"]) == 4
        assert res2.history["epoch"][-1] == 4
