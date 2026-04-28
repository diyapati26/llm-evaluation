from datasets import load_dataset
from evals.schemas import EvalSample, TaskType

def load_mmlu(subject: str = "all", split: str = "test", 
              max_samples: int = 20) -> list[EvalSample]:
    """
    Load MMLU benchmark questions.
    57 subjects — from high school math to professional medicine.
    Each question is 4-choice multiple choice.
    """
    ds = load_dataset(
        "cais/mmlu", subject, 
        split=split, 
        # trust_remote_code=True
    )
    
    samples = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        
        # Format the 4 choices into the prompt
        choices_text = "\n".join([
            f"A) {row['choices'][0]}",
            f"B) {row['choices'][1]}",
            f"C) {row['choices'][2]}",
            f"D) {row['choices'][3]}",
        ])
        
        prompt = (
            f"Question: {row['question']}\n\n"
            f"{choices_text}\n\n"
            f"Answer with A, B, C, or D only. "
            f"Do not explain."
        )
        
        # MMLU stores correct answer as 0,1,2,3 — convert to A,B,C,D
        correct_letter = ["A", "B", "C", "D"][row["answer"]]
        
        samples.append(EvalSample(
            id=f"mmlu_{subject}_{i}",
            prompt=prompt,
            reference=correct_letter,
            choices=["A", "B", "C", "D"],
            correct_choice=row["answer"],
            task_type=TaskType.multiple_choice,
            metadata={
                "subject":  subject,
                "source":   "mmlu",
                "question": row["question"],
            }
        ))
    
    print(f"Loaded {len(samples)} MMLU samples (subject: {subject})")
    return samples


def load_truthfulqa(split: str = "validation", 
                    max_samples: int = 20) -> list[EvalSample]:
    """
    Load TruthfulQA benchmark.
    Open-ended questions designed to elicit hallucinations.
    Tests whether models are honest about what they don't know.
    """
    ds = load_dataset("truthful_qa", "generation", split=split)
    
    samples = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        
        prompt = (
            f"Answer the following question truthfully and concisely.\n\n"
            f"Question: {row['question']}\n"
            f"Answer:"
        )
        
        samples.append(EvalSample(
            id=f"truthfulqa_{i}",
            prompt=prompt,
            reference=row["best_answer"],
            task_type=TaskType.open_generation,
            metadata={
                "source":            "truthfulqa",
                "category":          row["category"],
                "correct_answers":   row["correct_answers"],
                "incorrect_answers": row["incorrect_answers"],
            }
        ))
    
    print(f"Loaded {len(samples)} TruthfulQA samples")
    return samples


def load_hellaswag(split: str = "validation", 
                   max_samples: int = 20) -> list[EvalSample]:
    """
    Load HellaSwag commonsense reasoning benchmark.
    Pick the most plausible sentence completion.
    Tests grounded language understanding.
    """
    ds = load_dataset("Rowan/hellaswag", split=split)
    
    samples = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        
        endings = "\n".join([
            f"A) {row['endings'][0]}",
            f"B) {row['endings'][1]}",
            f"C) {row['endings'][2]}",
            f"D) {row['endings'][3]}",
        ])
        
        prompt = (
            f"Choose the most plausible continuation.\n\n"
            f"Context: {row['ctx']}\n\n"
            f"{endings}\n\n"
            f"Answer with A, B, C, or D only. "
            f"Do not explain."
        )
        
        correct_idx = int(row["label"])
        correct_letter = ["A", "B", "C", "D"][correct_idx]
        
        samples.append(EvalSample(
            id=f"hellaswag_{i}",
            prompt=prompt,
            reference=correct_letter,
            choices=["A", "B", "C", "D"],
            correct_choice=correct_idx,
            task_type=TaskType.multiple_choice,
            metadata={
                "source":         "hellaswag",
                "activity_label": row["activity_label"],
            }
        ))
    
    print(f"Loaded {len(samples)} HellaSwag samples")
    return samples


def load_all(max_per_dataset: int = 20) -> list[EvalSample]:
    """
    Convenience function — loads all three datasets at once.
    Default 20 samples each = 60 total.
    Perfect for dev runs.
    """
    return (
        load_mmlu(max_samples=max_per_dataset) +
        load_truthfulqa(max_samples=max_per_dataset) +
        load_hellaswag(max_samples=max_per_dataset)
    )

def load_mmlu_hard(
    subjects: list[str] = None,
    split: str = "test",
    max_per_subject: int = 5,
) -> list[EvalSample]:
    """
    Load MMLU questions from harder subjects specifically.
    Lower model confidence = more meaningful manipulation tests.
    Subjects chosen for ambiguity and domain specificity.
    """
    from evals.datasets.manipulation_dataset import HARD_SUBJECTS

    if subjects is None:
        subjects = HARD_SUBJECTS

    all_samples = []

    for subject in subjects:
        try:
            ds = load_dataset(
                "cais/mmlu",
                subject,
                split=split,
            )

            count = 0
            for i, row in enumerate(ds):
                if count >= max_per_subject:
                    break

                choices_text = "\n".join([
                    f"A) {row['choices'][0]}",
                    f"B) {row['choices'][1]}",
                    f"C) {row['choices'][2]}",
                    f"D) {row['choices'][3]}",
                ])

                prompt = (
                    f"Question: {row['question']}\n\n"
                    f"{choices_text}\n\n"
                    f"Answer with A, B, C, or D only. "
                    f"Do not explain."
                )

                correct_letter = ["A", "B", "C", "D"][row["answer"]]

                all_samples.append(EvalSample(
                    id=f"mmlu_{subject}_{i}",
                    prompt=prompt,
                    reference=correct_letter,
                    choices=["A", "B", "C", "D"],
                    correct_choice=row["answer"],
                    task_type=TaskType.multiple_choice,
                    metadata={
                        "subject":   subject,
                        "source":    "mmlu_hard",
                        "question":  row["question"],
                        "difficulty": "hard",
                    }
                ))
                count += 1

        except Exception as e:
            print(f"  Could not load {subject}: {e}")
            continue

    print(
        f"Loaded {len(all_samples)} hard MMLU samples "
        f"from {len(subjects)} subjects "
        f"({max_per_subject} per subject)"
    )
    return all_samples