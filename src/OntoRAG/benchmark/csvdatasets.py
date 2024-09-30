"""DSPy datasets for evaluation."""

import pandas as pd
from dspy.datasets.dataset import Dataset

answer_col = {
    "mmlumed": lambda x: x["answer"],
    "medqa": lambda x: x["answer_idx"],
    "medmcqa": lambda x: x["answer"].replace(
        {i + 1: v for i, v in enumerate(["A", "B", "C", "D"])}
    ),
}


class CSVDataset(Dataset):
    def __init__(self, file_path, dfname, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        df = pd.read_csv(file_path)
        df["answer"] = answer_col[dfname](df)
        self._dev = df.to_dict(orient="records")
        self.name = dfname
