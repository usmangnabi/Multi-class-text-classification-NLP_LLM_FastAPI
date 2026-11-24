# Loads the preprocessed dataset (created by notebooks/01_data_preprocessing.ipynb)
from functools import lru_cache  # lru_cache remembers results so we load only once

import pandas as pd  # pandas reads CSV files

from app.config import DATA_PATH, TRAIN_PATH  # file locations
from app.services.artifacts import require_file  # file check


@lru_cache  # run only once and reuse the result
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(require_file(DATA_PATH))  # table with text, clean_text, label


@lru_cache  # run only once and reuse the result
def load_train() -> pd.DataFrame:
    return pd.read_csv(require_file(TRAIN_PATH))  # training table


def get_labels() -> list:
    return sorted(load_dataset()["label"].unique().tolist())  # list of class names
