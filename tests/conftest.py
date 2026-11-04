import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from pulse.config import Settings  # noqa: E402


@pytest.fixture
def meta_settings() -> Settings:
    return Settings(backend="meta", base_url="https://api.meta.ai/v1", api_key="test-key", model="muse-spark-1.3",
                    reasoning_effort="low", recheck_reasoning_effort="high", batch_size=5, max_workers=2,
                    cache_path=":memory:")


@pytest.fixture(scope="session")
def survey() -> pd.DataFrame:
    from pulse.pipeline import load_survey

    return load_survey(ROOT / "data" / "synthetic_survey.csv")


@pytest.fixture(scope="session")
def gold_full() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "synthetic_labels_full.csv", dtype=str, keep_default_na=False)


@pytest.fixture(scope="session")
def gold_sample():
    from pulse.validation import load_gold

    return load_gold(ROOT / "data" / "human_coded_sample.csv")
