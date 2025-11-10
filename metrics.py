from functools import lru_cache
from bert_score import score, BERTScorer
import threading
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from constants import ground_truth_summary, ground_truth, ground_truth_facts
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

def nli_entailment_scores(premise: str, hypotheses: List[str], *, model_name: str = _DEFAULT_NLI_MODEL, batch_size: int = 16) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tokenizer, model, device = _load_nli(model_name)
    if not hypotheses:
        return np.array([]), np.array([]), np.array([])
    entail_probs, neutral_probs, contra_probs = [], [], []
    with torch.no_grad():
        for i in range(0, len(hypotheses), batch_size):
            batch_hyps = hypotheses[i:i+batch_size]
            enc = tokenizer(
                [premise] * len(batch_hyps),
                batch_hyps,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            contra = probs[:, 0]
            neutral = probs[:, 1]
            entail = probs[:, 2]
            entail_probs.extend(entail.tolist())
            neutral_probs.extend(neutral.tolist())
            contra_probs.extend(contra.tolist())
    return np.array(entail_probs), np.array(neutral_probs), np.array(contra_probs)

def compute_nli_fact_score(agent_summary: str, ground_truth_text: str, *, lambda_contra: float = 0.5, model_name: str = _DEFAULT_NLI_MODEL) -> dict:
    facts = split_into_atomic_facts(ground_truth_text)
    return compute_nli_fact_score_from_facts(agent_summary, facts, lambda_contra=lambda_contra, model_name=model_name)

def compute_nli_fact_score_from_facts(agent_summary: str, facts: List[str], *, lambda_contra: float = 0.5, model_name: str = _DEFAULT_NLI_MODEL) -> dict:
    if not facts or not agent_summary or agent_summary.strip() == "":
        return {
            "entail_mean": 0.0,
            "contradict_mean": 0.0,
            "neutral_mean": 0.0,
            "score": 0.0,
            "n_facts": len(facts) if facts else 0,
            "per_fact": [],
        }
    p_ent, p_neu, p_con = nli_entailment_scores(agent_summary, facts, model_name=model_name)
    entail_mean = float(p_ent.mean())
    contra_mean = float(p_con.mean())
    neutral_mean = float(p_neu.mean())
    combined = float(entail_mean - lambda_contra * contra_mean)
    per_fact = [
        {"fact": f, "p_entail": float(e), "p_contra": float(c), "p_neutral": float(n)}
        for f, e, c, n in zip(facts, p_ent, p_con, p_neu)
    ]
    return {
        "entail_mean": entail_mean,
        "contradict_mean": contra_mean,
        "neutral_mean": neutral_mean,
        "score": combined,
        "n_facts": len(facts),
        "per_fact": per_fact,
    }

def compute_nli_alignment_f1(agent_summary: str, ground_truth_text: str, *, entail_threshold: float = 0.5, model_name: str = _DEFAULT_NLI_MODEL) -> dict:
    """
    Bidirectional alignment using NLI:
      - Precision: fraction of agent facts that are entailed by (aligned with) any GT fact
      - Recall:    fraction of GT facts that are entailed by (aligned with) any agent fact
      - F1:        harmonic mean of precision and recall
    
    This evaluates fact-to-fact alignment rather than summary-to-fact, ensuring
    we measure what information the agent actually extracted vs. what it should have.
    """
    gt_facts = split_into_atomic_facts(ground_truth_text)
    agent_facts = split_into_atomic_facts(agent_summary)

    if not gt_facts and not agent_facts:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "n_agent_facts": 0, "n_gt_facts": 0}
    if not gt_facts:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "n_agent_facts": len(agent_facts), "n_gt_facts": 0}
    if not agent_facts:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0, "n_agent_facts": 0, "n_gt_facts": len(gt_facts)}

    # Recall: For each GT fact, check if ANY agent fact entails it
    # This measures: "Of the GT facts, how many did the agent capture?"
    # We check: "Does agent_fact_i entail gt_fact?" for each (agent_fact, gt_fact) pair
    gt_captured = []
    for gt_fact in gt_facts:
        # Check if any agent fact entails this GT fact
        # nli_entailment_scores(premise, [hypothesis]) checks: "Does premise entail hypothesis?"
        captured = False
        for agent_fact in agent_facts:
            p_ent, _, _ = nli_entailment_scores(agent_fact, [gt_fact], model_name=model_name, batch_size=16)
            if len(p_ent) > 0 and p_ent[0] >= entail_threshold:
                captured = True
                break
        gt_captured.append(captured)
    recall = float(sum(gt_captured) / len(gt_facts)) if gt_facts else 0.0

    # Precision: For each agent fact, check if ANY GT fact entails it
    # This measures: "Of the agent facts, how many are faithful (not hallucinated)?"
    # We check: "Does gt_fact_j entail agent_fact_i?" for each (gt_fact, agent_fact) pair
    agent_faithful = []
    for agent_fact in agent_facts:
        # Check if any GT fact entails this agent fact
        faithful = False
        for gt_fact in gt_facts:
            p_ent, _, _ = nli_entailment_scores(gt_fact, [agent_fact], model_name=model_name, batch_size=16)
            if len(p_ent) > 0 and p_ent[0] >= entail_threshold:
                faithful = True
                break
        agent_faithful.append(faithful)
    precision = float(sum(agent_faithful) / len(agent_facts)) if agent_facts else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_agent_facts": len(agent_facts),
        "n_gt_facts": len(gt_facts),
    }



def compute_score(agent_summary: str, gt: str, threshold: float = 0.5) -> dict:
    """
    Compute how much of the ground truth meaning is captured by the agent's summary.

    Parameters
    ----------
    agent_summary : str
        The text produced by the agent.
    ground_truth : str
        The original reference text.
    threshold : float, optional
        Similarity threshold to count a fact as 'captured' (default: 0.7).

    Returns
    -------
    dict
        {
            'average_raw_score': float,   # smooth semantic overlap (0-1)
            'recall_score': float,        # fraction of facts captured (0-1)
            'best_scores_per_fact': list, # similarity per ground truth sentence
        }
    """
    # --- 1. Split into sentences ---
    gt_sents = ground_truth_facts
    summary_sents = sent_tokenize(agent_summary)

    if len(gt_sents) == 0 or len(summary_sents) == 0:
        return {'average_raw_score': 0.0, 'recall_score': 0.0, 'best_scores_per_fact': []}

    # print(f"GT sentences: {gt_sents}")
    # print(f"Summary sentences: {summary_sents}")

    # --- 2. Encode all sentences ---
    gt_embeds = _model.encode(gt_sents, convert_to_tensor=True, show_progress_bar=False)
    summary_embeds = _model.encode(summary_sents, convert_to_tensor=True, show_progress_bar=False)
    

    # --- 3. Compute pairwise cosine similarity ---
    sim_matrix = util.cos_sim(summary_embeds, gt_embeds)  # shape [n_summary, n_gt]
    
    # Print matrix nicely with row/col headers
    # row_labels = [f"Summary {i+1}" for i in range(sim_matrix.size(0))]
    # col_labels = [f"Fact {j+1}" for j in range(sim_matrix.size(1))]
    
    # # Print header row
    # header = " " * 13 + "".join([f"{label:>12}" for label in col_labels])
    # print(header)
    # for i, row in enumerate(sim_matrix):
    #     row_str = f"{row_labels[i]:<12}"
    #     for val in row:
    #         row_str += f"{val.item():12.3f}"
    #     print(row_str)

    # --- 4. For each ground truth fact, find best match from summary ---
    best_scores_per_fact, _ = torch.max(sim_matrix, dim=0)

    # --- 5. Compute metrics ---
    average_raw_score = torch.mean(best_scores_per_fact).item()
    recall_score = torch.sum(best_scores_per_fact > threshold).item() / len(gt_sents)

    # return {
    #     "average_raw_score": round(average_raw_score, 3),
    #     "recall_score": round(recall_score, 3),
    #     "best_scores_per_fact": [round(x, 3) for x in best_scores_per_fact.tolist()],
    # }
    return average_raw_score


# ground_truth = """On a sunny Saturday, families explore the city zoo. Children rush to watch the lions being fed, while others gather by the pond where ducks splash and a little girl tosses crumbs. A vendor serves ice cream near a bench where an elderly couple enjoys the parrots’ chatter. Inside the humid reptile house, students sketch snakes for a biology project. A loudspeaker announces the upcoming penguin show, and crowds head toward the stadium. A child’s balloon drifts into a tree as laughter fills the air, blending with animal calls and the scent of popcorn."""

# agent_summary = """On a sunny Saturday morning at the city zoo, families and an elderly couple observed parrots and reptiles in the reptile area, while children approached the lion enclosure. A loudspeaker announced the upcoming penguin show scheduled to begin in ten minutes, with animal sounds and background laughter present."""

# score = compute_score(agent_summary, ground_truth)
# print(score)
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import numpy as np
import torch



# ---- load model once globally ----
_model = SentenceTransformer("all-mpnet-base-v2")

# ---- helper: semantic cosine average (0–1) ----
def _semantic_score(gt_sents, sum_sents):
    if not gt_sents or not sum_sents:
        return 0.0
    gt_emb = _model.encode(ground_truth_facts, convert_to_tensor=True, show_progress_bar=False)
    sm_emb = _model.encode(sum_sents, convert_to_tensor=True, show_progress_bar=False)
    cos = util.cos_sim(sm_emb, gt_emb)                       # [n_sum, n_gt]
    best_per_gt, _ = cos.max(dim=0)                          # [n_gt]
    best_per_gt01 = (best_per_gt.clamp(-1, 1) + 1.0) / 2.0   # map [-1,1]→[0,1]
    return float(best_per_gt01.mean().item())

# ---- helper: lexical BM25 average normalized (0–1) ----
def _bm25_score(gt_sents, sum_sents):
    if not gt_sents or not sum_sents:
        return 0.0
    token_sum = [word_tokenize(s.lower()) for s in sum_sents]
    token_gt  = [word_tokenize(s.lower()) for s in ground_truth_facts]
    bm25 = BM25Okapi(token_sum)
    best = []
    for q in token_gt:
        scores = bm25.get_scores(q)
        best.append(float(np.max(scores)) if len(scores) else 0.0)
    if not best:
        return 0.0
    # Map raw BM25 scores into (0, 1) for compatibility with the combined metric.
    normalized = [1.0 - float(np.exp(-score)) for score in best]
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
    print("--------------------------------")
    print(f"Computing the final score with the following: {ground_truth_facts}")
    print(f"And the following: {sum_sents}")
    print(f"The semantic score is: {semantic}")
    print(f"The lexical score is: {lexical}")
    print(f"The final score is: {final_score}")
    print("--------------------------------")

    # return {
    #     "semantic_score": round(semantic, 3),
    #     "lexical_score": round(lexical, 3),
    #     "final_score": round(final_score, 3),
    # }
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