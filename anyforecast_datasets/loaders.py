import dataclasses
import os
from typing import Protocol

import pandas as pd

from anyforecast_datasets import definitions, readers


def get_filepath(filename: str) -> str:
    return os.path.join(definitions.DATA_DIR, filename)


class Reader(Protocol):

    def read(self, filepath: str) -> pd.DataFrame: ...


@dataclasses.dataclass
class TimeseriesDataset:
    target: list[str]
    group_cols: list[str]
    datetime: str
    feature_names: list[str]
    freq: str
    filepath: str
    reader: Reader

    def read(self):
        return self.reader.read(self.filepath)


@dataclasses.dataclass
class Dataset:
    target: list[str]
    feature_names: list[str]
    filepath: str
    reader: Reader

    def read(self):
        return self.reader.read(self.filepath)


def load_stallion() -> TimeseriesDataset:
    """Loads and returns the iris dataset (time series)."""

    filepath = get_filepath("stallion.csv")

    feature_names = [
        "agency",
        "sku",
        "date",
        "industry_volume",
        "price_regular",
        "price_actual",
        "discount",
    ]

    target = "volume"
    group_cols = ["agency", "sku"]
    datetime = "date"
    freq = "MS"

    return TimeseriesDataset(
        target=target,
        group_cols=group_cols,
        datetime=datetime,
        freq=freq,
        feature_names=feature_names,
        filepath=filepath,
        reader=readers.CSVReader(),
    )


def load_iris() -> Dataset:
    """Loads and returns the iris dataset (classification)."""

    filepath = get_filepath("iris.csv")

    target = "species"

    feature_names = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]

    return Dataset(
        target=target,
        feature_names=feature_names,
        filepath=filepath,
        reader=readers.CSVReader(),
    )
