"""
data_utils.py
Reusable data loading and preprocessing functions for HR Analytics project.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_raw_data(data_dir: str = "data/raw") -> tuple:
    """Load raw train and test CSVs. Returns (train, test) DataFrames."""
    path = Path(data_dir)
    train = pd.read_csv(path / "aug_train.csv")
    test  = pd.read_csv(path / "aug_test.csv")
    print(f"Train shape: {train.shape}")
    print(f"Test shape:  {test.shape}")
    return train, test


def get_missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted DataFrame of missing counts and percentages."""
    report = pd.DataFrame({
        'missing_count': df.isnull().sum(),
        'missing_pct':   (df.isnull().sum() / len(df) * 100).round(2)
    }).sort_values('missing_pct', ascending=False)
    return report[report['missing_count'] > 0]


def get_target_distribution(df: pd.DataFrame,
                             target_col: str = 'target') -> pd.DataFrame:
    """Return class counts and percentages for the target column."""
    counts = df[target_col].value_counts()
    pcts   = df[target_col].value_counts(normalize=True).round(4) * 100
    return pd.DataFrame({'count': counts, 'pct': pcts})


def encode_experience(series: pd.Series) -> pd.Series:
    """
    Map experience strings to ordinal integers.
    '<1' -> 0, '1'..'20' -> 1..20, '>20' -> 21, NaN -> NaN
    """
    exp_map = {'<1': 0, '>20': 21, **{str(i): i for i in range(1, 21)}}
    return series.map(exp_map)


def encode_last_new_job(series: pd.Series) -> pd.Series:
    """
    Map last_new_job strings to ordinal integers.
    'never' -> 0, '1'..'4' -> 1..4, '>4' -> 5, NaN -> NaN
    """
    lnj_map = {'never': 0, '>4': 5, **{str(i): i for i in range(1, 5)}}
    return series.map(lnj_map)


def encode_company_size(series: pd.Series) -> pd.Series:
    """
    Map company_size bands to ordinal integers 1-8.
    Unknown/NaN -> 0 (sentinel for missing).
    Note: '10/49' treated as '10-49' (formatting inconsistency in source data).
    """
    size_map = {
        '<10':       1,
        '10/49':     2,
        '10-49':     2,
        '50-99':     3,
        '100-500':   4,
        '500-999':   5,
        '1000-4999': 6,
        '5000-9999': 7,
        '10000+':    8
    }
    return series.map(size_map).fillna(0)


def encode_education(series: pd.Series) -> pd.Series:
    """
    Map education_level to ordinal integers 1-5.
    NaN -> NaN (handled separately with median imputation).
    """
    edu_map = {
        'Primary School': 1,
        'High School':    2,
        'Graduate':       3,
        'Masters':        4,
        'Phd':            5
    }
    return series.map(edu_map)


def add_missing_indicators(df_target: pd.DataFrame,
                            df_raw: pd.DataFrame,
                            cols: list) -> pd.DataFrame:
    """
    Add binary flags indicating whether each column was originally missing.
    Must be computed from the raw DataFrame BEFORE imputation.
    """
    df = df_target.copy()
    for col in cols:
        if col in df_raw.columns:
            df[f'{col}_was_missing'] = df_raw[col].isnull().astype(int)
    return df


def add_career_stage(df: pd.DataFrame,
                     exp_col: str = 'experience_num') -> pd.DataFrame:
    """
    Add career_stage_num: Early=0 (0-3 yrs), Mid=1 (4-10), Senior=2 (11+).
    Rationale: EDA showed non-linear attrition arc peaking at early stage.
    """
    def _stage(x):
        if pd.isna(x):  return np.nan
        if x <= 3:      return 0
        elif x <= 10:   return 1
        else:           return 2

    df = df.copy()
    df['career_stage_num'] = df[exp_col].apply(_stage)
    return df


def add_stability_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add stability_score: normalized average of company_size_num and last_new_job_num.
    Bounded [0, 1]. Large stable company + long tenure = high score = low risk.
    """
    df = df.copy()
    df['stability_score'] = (
        df['company_size_num'] / 8 +
        df['last_new_job_num'] / 5
    ) / 2
    return df


def add_upskilling_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add upskilling_intensity: log(training_hours) normalized by experience.
    Captures relative training investment controlling for career stage.
    """
    df = df.copy()
    df['upskilling_intensity'] = (
        df['training_hours_log'] / 
        (df['experience_num'] + 1)
    )
    return df


def validate_preprocessed(df_train: pd.DataFrame,
                           df_test:  pd.DataFrame) -> bool:
    """
    Run sanity checks on the preprocessed DataFrames.
    Raises AssertionError with a descriptive message on failure.
    Returns True if all checks pass.
    """
    print("=" * 55)
    print("PREPROCESSING VALIDATION REPORT")
    print("=" * 55)

    # 1. No missing values
    assert df_train.isnull().sum().sum() == 0, \
        f"FAIL: {df_train.isnull().sum().sum()} missing values in train"
    assert df_test.isnull().sum().sum() == 0, \
        f"FAIL: {df_test.isnull().sum().sum()} missing values in test"
    print("✓ No missing values in train or test")

    # 2. All columns numeric (allow bool from get_dummies)
    non_num = [c for c in df_train.columns
               if df_train[c].dtype not in ['float64','int64','int32','bool',
                                             'float32','uint8']]
    assert len(non_num) == 0, f"FAIL: Non-numeric columns: {non_num}"
    print("✓ All columns are numeric")

    # 3. Target column present in train, absent in test
    assert 'target' in df_train.columns,    "FAIL: target missing from train"
    assert 'target' not in df_test.columns, "FAIL: target present in test (leakage)"
    print("✓ Target column correct (present in train, absent in test)")

    # 4. Column alignment
    train_feats = set(df_train.columns) - {'target'}
    test_feats  = set(df_test.columns)
    only_train  = train_feats - test_feats
    only_test   = test_feats  - train_feats
    assert train_feats == test_feats, \
        f"FAIL: Column mismatch.\nTrain only: {only_train}\nTest only: {only_test}"
    print("✓ Train and test columns are aligned")

    # 5. No ID column
    assert 'enrollee_id' not in df_train.columns, \
        "FAIL: enrollee_id still present (potential leakage)"
    print("✓ No ID column present")

    # 6. Stability score bounded
    if 'stability_score' in df_train.columns:
        assert df_train['stability_score'].between(0, 1).all(), \
            "FAIL: stability_score outside [0, 1]"
        print("✓ stability_score bounded [0, 1]")

    print("=" * 55)
    print(f"Feature count : {len(train_feats)}")
    print(f"Training rows : {len(df_train):,}")
    print(f"Test rows     : {len(df_test):,}")
    print("=" * 55)
    print("ALL CHECKS PASSED - ready to save")
    return True
