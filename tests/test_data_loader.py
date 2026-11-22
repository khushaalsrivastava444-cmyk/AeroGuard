"""Unit tests for the NASA C-MAPSS FD001 data loader."""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.data.loader import (
    ALL_COLUMNS,
    INDEX_COLUMNS,
    SENSOR_COLUMNS,
    SETTING_COLUMNS,
    load_fd001_raw,
    load_fd001_rul,
    load_fd001_test,
    load_fd001_train,
)


def test_schema_definitions():
    """Verify constant column definitions and dimensional alignment."""
    assert len(INDEX_COLUMNS) == 2
    assert len(SETTING_COLUMNS) == 3
    assert len(SENSOR_COLUMNS) == 21
    assert len(ALL_COLUMNS) == 26
    assert ALL_COLUMNS[:2] == ["engine_id", "cycle"]
    assert ALL_COLUMNS[2:5] == ["setting_1", "setting_2", "setting_3"]
    assert ALL_COLUMNS[5:] == [f"s_{i}" for i in range(1, 22)]


def test_load_train_fd001():
    """Verify structural contracts for FD001 training trajectories."""
    train_df = load_fd001_train()

    assert isinstance(train_df, pd.DataFrame)
    assert train_df.shape == (20631, 26)
    assert list(train_df.columns) == ALL_COLUMNS
    assert train_df.isna().sum().sum() == 0

    # Engine IDs must range 1 to 100 with all engines present
    unique_engines = np.sort(train_df["engine_id"].unique())
    assert len(unique_engines) == 100
    assert unique_engines[0] == 1
    assert unique_engines[-1] == 100

    # Ensure integer types for identifiers and floats for measurements
    assert np.issubdtype(train_df["engine_id"].dtype, np.integer)
    assert np.issubdtype(train_df["cycle"].dtype, np.integer)
    for col in SETTING_COLUMNS + SENSOR_COLUMNS:
        assert np.issubdtype(train_df[col].dtype, np.floating)


def test_load_test_fd001():
    """Verify structural contracts for FD001 test trajectories."""
    test_df = load_fd001_test()

    assert isinstance(test_df, pd.DataFrame)
    assert test_df.shape == (13096, 26)
    assert list(test_df.columns) == ALL_COLUMNS
    assert test_df.isna().sum().sum() == 0

    unique_engines = np.sort(test_df["engine_id"].unique())
    assert len(unique_engines) == 100
    assert unique_engines[0] == 1
    assert unique_engines[-1] == 100


def test_load_rul_fd001():
    """Verify structural contracts for official test ground-truth RUL."""
    rul_series = load_fd001_rul()

    assert isinstance(rul_series, pd.Series)
    assert len(rul_series) == 100
    assert rul_series.isna().sum() == 0
    assert (rul_series > 0).all(), "Remaining Useful Life must be strictly positive"
    assert rul_series.index.min() == 1
    assert rul_series.index.max() == 100


def test_load_fd001_raw_bundle():
    """Verify combined loading utility consistency."""
    train_df, test_df, rul_series = load_fd001_raw()
    assert train_df.shape == (20631, 26)
    assert test_df.shape == (13096, 26)
    assert len(rul_series) == 100


def test_missing_file_raises_error(tmp_path: Path):
    """Verify loader raises FileNotFoundError when target files are absent."""
    with pytest.raises(FileNotFoundError):
        load_fd001_train(data_dir=tmp_path)
