import re
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn

class ExactMatchScorer:
    """
    Binary scorer — did the model get it exactly right?
    Handles both multiple choice (A/B/C/D) and short text answers.
    """
    name = "exact_match"

    def score(self, prediction: str, reference: str, **kwargs) -> dict:
        # Clean both strings — lowercase, strip whitespace and punctuation
        def clean(text: str) -> str:
            return re.sub(r'[^\w\s]', '', text.lower().strip())

        pred_clean = clean(prediction)
        ref_clean  = clean(reference)

        # Extract first letter for multiple choice
        # Model might say "A" or "A)" or "Answer: A" or "The answer is A"
        pred_letter = pred_clean[0] if pred_clean else ""
        ref_letter  = ref_clean[0]  if ref_clean  else ""

        exact       = float(pred_clean == ref_clean)
        letter_match = float(pred_letter == ref_letter)

        # Take the more generous match — full exact or first letter
        final_score = max(exact, letter_match)

        return {
            "score":        final_score,
            "exact":        exact,
            "letter_match": letter_match,
            "prediction":   prediction[:100],
            "reference":    reference[:100],
        }


class RougeScorer:
    """
    ROUGE-L scorer — measures longest common subsequence overlap.
    Good for open-ended answers where word order matters.
    Scores between 0 and 1.
    """
    name = "rouge_l"

    def __init__(self):
        self._scorer = rouge_scorer.RougeScorer(
            ["rougeL"], 
            use_stemmer=True  # "running" and "runs" count as a match
        )

    def score(self, prediction: str, reference: str, **kwargs) -> dict:
        if not prediction or not reference:
            return {"score": 0.0, "precision": 0.0, "recall": 0.0}

        result  = self._scorer.score(reference, prediction)
        rouge_l = result["rougeL"]

        return {
            "score":     rouge_l.fmeasure,
            "precision": rouge_l.precision,
            "recall":    rouge_l.recall,
        }


class BERTScorer:
    """
    BERTScore — semantic similarity using contextual embeddings.
    'Car' and 'automobile' score high even though different words.
    Slower than ROUGE but much better at capturing meaning.
    First run downloads a BERT model (~400MB) — cached after that.
    """
    name = "bertscore"

    def score(self, prediction: str, reference: str, **kwargs) -> dict:
        if not prediction or not reference:
            return {"score": 0.0, "precision": 0.0, "recall": 0.0}

        P, R, F1 = bert_score_fn(
            [prediction],
            [reference],
            lang="en",
            rescale_with_baseline=True,
            verbose=False
        )

        return {
            "score":     round(float(F1[0]), 4),
            "precision": round(float(P[0]),  4),
            "recall":    round(float(R[0]),  4),
        } 