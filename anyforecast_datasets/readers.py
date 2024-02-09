import pandas as pd


class CSVReader:

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def read(self, filepath) -> pd.DataFrame:
        return pd.read_csv(filepath, **self.kwargs)


class ParquetReader:

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def read(self, filepath) -> pd.DataFrame:
        return pd.read_parquet(filepath, **self.kwargs)
