from functools import lru_cache
from bert_score import score, BERTScorer
import threading
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
_model = SentenceTransformer("all-mpnet-base-v2")
from nltk.tokenize import sent_tokenize
import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# ground_truth_facts = sent_tokenize(ground_truth)

_scorer_lock = threading.Lock()
_scorer_cache = {}

@lru_cache(maxsize=10)
def get_scorer(model_type: str = "roberta-large", device: str | None = None, idf: bool = False, lang: str = "en") -> BERTScorer:
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

def compute_bert_score(summary: str, ground_truth: str, *, model_type: str = "roberta-large", lang: str = "en"):
    """Compute BERT score using a shared, cached scorer for thread safety"""
    candidates = [summary] if isinstance(summary, str) else summary
    references = [ground_truth] if isinstance(ground_truth, str) else ground_truth

    # Get cached scorer WITH lang parameter
    scorer = get_scorer(model_type=model_type, lang=lang)
    
    # Use the scorer's score method directly (not the module-level score function)
    P, R, F1 = scorer.score(candidates, references)
    return F1.mean().item()






# --- NLI-based fact entailment scoring ---
from typing import List, Tuple, Optional
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

_DEFAULT_NLI_MODEL = "roberta-large-mnli"

@lru_cache(maxsize=2)
def _load_nli(model_name: str = _DEFAULT_NLI_MODEL, device: Optional[str] = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device

_SENT_SPLIT_RE = re.compile(r"(?<=[\.!?])\s+(?=[A-Z0-9])")
_CLAUSE_SPLIT_RE = re.compile(r"\s*(?:,|;|:|\band\b|\bwhile\b|\bwhere\b|\bas\b)\s+", re.IGNORECASE)

def split_into_atomic_facts(text: str) -> List[str]:
    if not text:
        return []
    cleaned = text.strip()
    if not cleaned:
        return []
    sentences = re.split(_SENT_SPLIT_RE, cleaned)
    clauses: List[str] = []
    for sent in sentences:
        if not sent or not sent.strip():
            continue
        clauses.extend([c.strip() for c in re.split(_CLAUSE_SPLIT_RE, sent.strip()) if len(c.strip()) >= 3])
    return clauses

def split_sentences(text: str) -> List[str]:
    """Regex-based sentence tokenizer to avoid external dependencies."""
    if not text:
        return []
    cleaned = text.strip()
    if not cleaned:
        return []
    parts = re.split(_SENT_SPLIT_RE, cleaned)
    return [s.strip() for s in parts if len(s.strip()) >= 3]

def compute_score(agent_summary: str, gt: str, threshold: float = 0.5) -> dict:

    gt_sents = gt
    summary_sents = agent_summary

    if len(gt_sents) == 0 or len(summary_sents) == 0:
        return {'average_raw_score': 0.0, 'recall_score': 0.0, 'best_scores_per_fact': []}
    gt_embeds = _model.encode(gt_sents, convert_to_tensor=True, show_progress_bar=False)
    summary_embeds = _model.encode(summary_sents, convert_to_tensor=True, show_progress_bar=False)
    
    sim_matrix = util.cos_sim(summary_embeds, gt_embeds)  # shape [n_summary, n_gt]

    best_scores_per_fact, _ = torch.max(sim_matrix, dim=0)

    average_raw_score = torch.mean(best_scores_per_fact).item()
    recall_score = torch.sum(best_scores_per_fact > threshold).item() / len(gt_sents)
    return average_raw_score

from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import numpy as np
import torch



_model = SentenceTransformer("all-mpnet-base-v2")

def _semantic_score(gt_sents, sum_sents):
    if not gt_sents or not sum_sents:
        return 0.0
    gt_emb = _model.encode(gt_sents, convert_to_tensor=True, show_progress_bar=False)
    sm_emb = _model.encode(sum_sents, convert_to_tensor=True, show_progress_bar=False)
    cos = util.cos_sim(sm_emb, gt_emb)
    best_per_gt, _ = cos.max(dim=0)
    best_per_gt01 = (best_per_gt.clamp(-1, 1) + 1.0) / 2.0
    return float(best_per_gt01.mean().item())

def _bm25_score(gt_sents, sum_sents):
    if not gt_sents or not sum_sents:
        return 0.0
    token_sum = [word_tokenize(s.lower()) for s in sum_sents]
    token_gt  = [word_tokenize(s.lower()) for s in gt_sents]
    bm25 = BM25Okapi(token_sum)
    best = []
    for q in token_gt:
        scores = bm25.get_scores(q)
        best.append(float(np.max(scores)) if len(scores) else 0.0)
    if not best:
        return 0.0
    # Map raw BM25 scores into (0, 1) for compatixbility with the combined metric.
    normalized = [float(1.0 / (1.0 + np.exp(-score))) for score in best]
    return float(np.mean(normalized))

# ---- main combined scorer ----
def compute_final_score(agent_summary: str, ground_truth: str, alpha: float = 0.7) -> dict:
    """
    Returns a combined semantic + lexical similarity metric.

    alpha = weight on semantic similarity (default 0.7)
    final_score = alpha * semantic + (1 - alpha) * lexical
    """
    gt_sents  = [s.strip() for s in sent_tokenize(ground_truth) if s.strip()]
    sum_sents = [s.strip() for s in sent_tokenize(agent_summary) if s.strip()]
    
    # semantic = _semantic_score(gt_sents, sum_sents)
    semantic = compute_score(agent_summary, ground_truth)
    lexical  = _bm25_score(gt_sents, sum_sents)
    final_score = alpha * semantic + (1 - alpha) * lexical
    # print("--------------------------------")
    # print(f"Computing the final score with the following: {gt_sents}")
    # print(f"And the following: {sum_sents}")
    # print(f"The semantic score is: {semantic}")
    # print(f"The lexical score is: {lexical}")
    # print(f"The final score is: {final_score}")
    # print("--------------------------------")
    return final_score

# ---- quick test ----
if __name__ == "__main__":
    ground_truth = """Students meet recruiters from many companies. Recruiters offer internships and full-time jobs.
    Students hand out resumes and practice elevator pitches. Workshops and panels help students prepare for interviews.
    Career counselors guide students to suitable employers. The fair is energetic and full of opportunity."""

    summary = """Students prepared for career fairs by developing resumes, business cards, and elevator pitches.
    Recruiters from consulting, technology, and startup firms conducted pitch practice sessions and attended the fairs.
    University counselors facilitated small group discussions and distributed branded tote bags.
    Recruiters, university counselors, and career counselors connected students with startups and technology firms."""

    result = compute_final_score(summary, ground_truth)
    print(result)