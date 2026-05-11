"""Dataset loaders — all return pandas DataFrames sampled to size.

Unified entry point: load_dataset_by_name("mmlu" | "hellaswag" | "truthfulqa_mc" |
"truthfulqa_gen", **kwargs). Used by runner.py and any consumer that wants to
load by string name rather than importing a specific loader.
"""
from datasets import load_dataset  # HuggingFace
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


HARD_MMLU = [
    "professional_medicine",
    "international_law",
    "college_physics",
    "abstract_algebra",
    "professional_law",
    "college_chemistry",
    "high_school_statistics",
    "machine_learning",
]


def load_dataset_by_name(
    name: Literal["mmlu", "hellaswag", "truthfulqa_mc", "truthfulqa_gen", "hard_mmlu"],
    n: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """Single entry point: load any supported dataset by string name.

    `n` is total samples for hellaswag / truthfulqa, and samples-per-subject for
    mmlu / hard_mmlu (divided across the subject list).

    Returns a DataFrame with a "subject" column populated for grouping.
    """
    if name == "mmlu":
        df = load_mmlu_data_sample(subjects="all", max_per_subject=n, random_state=random_state)
        return df

    if name == "hard_mmlu":
        df = load_mmlu_data_sample(
            subjects=HARD_MMLU,
            max_per_subject=max(1, n // len(HARD_MMLU)),
            random_state=random_state,
        )
        return df

    if name == "hellaswag":
        df = load_hellaswag_data_sample(max_samples=n, random_state=random_state)
        df["subject"] = "hellaswag"
        return df

    if name == "truthfulqa_mc":
        df = load_truthfulqa_data_sample(config="multiple_choice", max_samples=n, random_state=random_state)
        df["subject"] = "truthfulqa_mc"
        return df

    if name == "truthfulqa_gen":
        df = load_truthfulqa_data_sample(config="generation", max_samples=n, random_state=random_state)
        df["subject"] = "truthfulqa_gen"
        return df

    raise ValueError(
        f"Unknown dataset '{name}'. "
        f"Choose from: mmlu, hard_mmlu, hellaswag, truthfulqa_mc, truthfulqa_gen."
    )


def load_all(n_per_dataset: int = 10) -> dict[str, pd.DataFrame]:
    """Load every supported dataset, return a {name: dataframe} dict.

    Useful for cross-dataset comparisons or one-shot full-suite runs.
    """
    return {
        "mmlu":          load_dataset_by_name("hard_mmlu",      n=n_per_dataset),
        "hellaswag":     load_dataset_by_name("hellaswag",      n=n_per_dataset),
        "truthfulqa_mc": load_dataset_by_name("truthfulqa_mc",  n=n_per_dataset),
        "truthfulqa_gen": load_dataset_by_name("truthfulqa_gen", n=n_per_dataset),
    }