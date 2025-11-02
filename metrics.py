from functools import lru_cache
from bert_score import score, BERTScorer
import threading

_scorer_lock = threading.Lock()
_scorer_cache = {}

@lru_cache(maxsize=10)
def get_scorer(model_type: str = "bert-base-uncased", device: str | None = None, idf: bool = False, lang: str = "en") -> BERTScorer:
    """Get or create a cached BERTScorer instance with thread safety"""
    # Include lang in cache key
    cache_key = (model_type, device, idf, lang)
    with _scorer_lock:
        if cache_key not in _scorer_cache:
            _scorer_cache[cache_key] = BERTScorer(
                model_type=model_type, 
                device=device, 
                idf=idf, 
                rescale_with_baseline=True,
                lang=lang  # Pass lang parameter here!
            )
        return _scorer_cache[cache_key]

def compute_bert_score(summary: str, ground_truth: str, *, model_type: str = "bert-base-uncased", lang: str = "en"):
    """Compute BERT score using a shared, cached scorer for thread safety"""
    candidates = [summary] if isinstance(summary, str) else summary
    references = [ground_truth] if isinstance(ground_truth, str) else ground_truth

    # Get cached scorer WITH lang parameter
    scorer = get_scorer(model_type=model_type, lang=lang)
    
    # Use the scorer's score method directly (not the module-level score function)
    P, R, F1 = scorer.score(candidates, references)
    return F1.mean().item()
