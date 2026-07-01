"""Shared orchestration helpers for training pipelines."""

from .transformer_training import (
    build_dataset,
    build_datasets,
    fit_cca_and_emg_stats,
    make_preprocess_fn,
    prepare_batch_factory,
    unique_series_arrays,
)

__all__ = [
    "make_preprocess_fn",
    "build_dataset",
    "build_datasets",
    "unique_series_arrays",
    "fit_cca_and_emg_stats",
    "prepare_batch_factory",
]
