from datasets import load_dataset #HuggingFace
from typing import Literal, Union
import pandas as pd

def load_mmlu_data_sample(
    subjects: Union[list[str], Literal["all"]] = "all",
    split: Literal["test", "validation", "dev", "auxiliary_train"] = "test",
    max_per_subject: int = 20,
    random_state: int = 42,) -> pd.DataFrame:
    
    if subjects == "all":
        ds = load_dataset("cais/mmlu", "all", split=split)
        df = ds.to_pandas()
    else:
        dfs = []
        for subject in subjects:
            ds = load_dataset("cais/mmlu", subject, split=split)
            subject_df = ds.to_pandas()
            subject_df["subject"] = subject
            dfs.append(subject_df)
        df = pd.concat(dfs, ignore_index=True)
    sample = (
        df.groupby("subject", group_keys=False)
          .sample(n=max_per_subject, random_state=random_state)
          .reset_index(drop=True)
    )
    print(sample["subject"].value_counts())
    return sample

def load_truthfulqa_data_sample(
    config: Literal["generation", "multiple_choice"] = "multiple_choice",
    max_samples: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    # TruthfulQA only has a 'validation' split (817 questions total)
    ds = load_dataset("truthful_qa", config, split="validation")
    df = ds.to_pandas()
    sample = df.sample(n=max_samples, random_state=random_state).reset_index(drop=True)
    return sample


def load_hellaswag_data_sample(
    split: Literal["train", "validation", "test"] = "validation",
    max_samples: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    ds = load_dataset("Rowan/hellaswag", split=split)
    df = ds.to_pandas()
    sample = df.sample(n=max_samples, random_state=random_state).reset_index(drop=True)
    return sample