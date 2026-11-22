"""NASA C-MAPSS FD001 Dataset Loader.

Provides deterministic data ingestion, column schema mapping, and contract
verification for the NASA C-MAPSS Turbofan Engine Degradation Simulation dataset.
"""

from pathlib import Path
from typing import Final, Tuple, Union

import pandas as pd

# Schema definitions per NASA C-MAPSS documentation
INDEX_COLUMNS: Final[list[str]] = ["engine_id", "cycle"]
SETTING_COLUMNS: Final[list[str]] = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS: Final[list[str]] = [f"s_{i}" for i in range(1, 22)]
ALL_COLUMNS: Final[list[str]] = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS

# Canonical C-MAPSS sensor descriptions (Saxena et al., 2008, Table 2)
SENSOR_METADATA: Final[dict[str, dict[str, str]]] = {
    "s_1": {"symbol": "T2", "description": "Total temperature at fan inlet", "units": "\u00b0R"},
    "s_2": {"symbol": "T24", "description": "Total temperature at LPC outlet", "units": "\u00b0R"},
    "s_3": {"symbol": "T30", "description": "Total temperature at HPC outlet", "units": "\u00b0R"},
    "s_4": {"symbol": "T50", "description": "Total temperature at LPT outlet", "units": "\u00b0R"},
    "s_5": {"symbol": "P2", "description": "Pressure at fan inlet", "units": "psia"},
    "s_6": {"symbol": "P15", "description": "Total pressure in bypass-duct", "units": "psia"},
    "s_7": {"symbol": "P30", "description": "Total pressure at HPC outlet", "units": "psia"},
    "s_8": {"symbol": "Nf", "description": "Physical fan speed", "units": "rpm"},
    "s_9": {"symbol": "Nc", "description": "Physical core speed", "units": "rpm"},
    "s_10": {"symbol": "epr", "description": "Engine pressure ratio (P50/P2)", "units": "--"},
    "s_11": {"symbol": "Ps30", "description": "Static pressure at HPC outlet", "units": "psia"},
    "s_12": {"symbol": "phi", "description": "Ratio of fuel flow to Ps30", "units": "pps/psi"},
    "s_13": {"symbol": "NRf", "description": "Corrected fan speed", "units": "rpm"},
    "s_14": {"symbol": "NRc", "description": "Corrected core speed", "units": "rpm"},
    "s_15": {"symbol": "BPR", "description": "Bypass Ratio", "units": "--"},
    "s_16": {"symbol": "farB", "description": "Burner fuel-air ratio", "units": "--"},
    "s_17": {"symbol": "htBleed", "description": "Bleed Enthalpy", "units": "--"},
    "s_18": {"symbol": "Nf_dmd", "description": "Demanded fan speed", "units": "rpm"},
    "s_19": {"symbol": "PCNfR_dmd", "description": "Demanded corrected fan speed", "units": "rpm"},
    "s_20": {"symbol": "W31", "description": "HPT coolant bleed", "units": "lbm/s"},
    "s_21": {"symbol": "W32", "description": "LPT coolant bleed", "units": "lbm/s"},
}

# Explicit type mapping to prevent accidental type coercion
DTYPES: Final[dict[str, Union[type, str]]] = {
    "engine_id": int,
    "cycle": int,
    **{col: float for col in SETTING_COLUMNS},
    **{col: float for col in SENSOR_COLUMNS},
}

DEFAULT_RAW_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "data" / "raw"


def _validate_trajectory_df(df: pd.DataFrame, dataset_name: str) -> None:
    """Validate structural integrity of an ingested trajectory dataframe."""
    if df.shape[1] != len(ALL_COLUMNS):
        raise ValueError(
            f"Expected {len(ALL_COLUMNS)} columns for {dataset_name}, got {df.shape[1]}"
        )

    null_count = df.isna().sum().sum()
    if null_count > 0:
        raise ValueError(f"{dataset_name} contains {null_count} unexpected null values")

    # Verify cycle starts at 1 and increases monotonically by 1 per engine
    grouped = df.groupby("engine_id")["cycle"]
    for engine_id, cycles in grouped:
        cycle_series = cycles.to_numpy()
        if cycle_series[0] != 1:
            raise ValueError(
                f"{dataset_name}: Engine {engine_id} does not start at cycle 1 (starts at {cycle_series[0]})"
            )
        diffs = cycle_series[1:] - cycle_series[:-1]
        if len(diffs) > 0 and (diffs != 1).any():
            raise ValueError(
                f"{dataset_name}: Engine {engine_id} cycles are not strictly monotonic step-1 increments"
            )


def load_fd001_train(data_dir: Union[str, Path] = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Load and validate NASA C-MAPSS FD001 training trajectory data.

    Returns:
        pd.DataFrame: 20,631 rows, 26 columns across 100 run-to-failure engines.
    """
    path = Path(data_dir) / "train_FD001.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Training dataset not found at: {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=ALL_COLUMNS,
        dtype=DTYPES,
    )
    _validate_trajectory_df(df, "train_FD001")
    return df


def load_fd001_test(data_dir: Union[str, Path] = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Load and validate NASA C-MAPSS FD001 test trajectory data.

    Returns:
        pd.DataFrame: 13,096 rows, 26 columns across 100 truncated engines.
    """
    path = Path(data_dir) / "test_FD001.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Test dataset not found at: {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=ALL_COLUMNS,
        dtype=DTYPES,
    )
    _validate_trajectory_df(df, "test_FD001")
    return df


def load_fd001_rul(data_dir: Union[str, Path] = DEFAULT_RAW_DIR) -> pd.Series:
    """Load ground-truth Remaining Useful Life (RUL) for test engines at their last observed cycle.

    Returns:
        pd.Series: 100 integer RUL values indexed by engine_id (1 to 100).
    """
    path = Path(data_dir) / "RUL_FD001.txt"
    if not path.is_file():
        raise FileNotFoundError(f"RUL ground-truth file not found at: {path}")

    series = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["rul"],
        dtype=int,
    )["rul"]

    if len(series) != 100:
        raise ValueError(f"Expected 100 ground-truth RUL entries, found {len(series)}")

    # Index 1-based to match engine_id
    series.index = pd.RangeIndex(start=1, stop=101, name="engine_id")
    return series


def load_fd001_raw(
    data_dir: Union[str, Path] = DEFAULT_RAW_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the complete FD001 bundle: (train_df, test_df, rul_series)."""
    train_df = load_fd001_train(data_dir)
    test_df = load_fd001_test(data_dir)
    rul_series = load_fd001_rul(data_dir)
    return train_df, test_df, rul_series
