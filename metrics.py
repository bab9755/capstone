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

    best_scores_per_fact, _ = torch.max(sim_matrix, dim=1)

    average_raw_score = torch.mean(best_scores_per_fact).item()
    recall_score = torch.sum(best_scores_per_fact > threshold).item() / len(gt_sents)
    return average_raw_score


def compute_similarity_matrix(agent_summary: str, gt_facts) -> dict:
    """
    Compute the full cosine similarity matrix between each sentence in the
    agent's summary and each ground-truth fact.

    Returns a dict with:
      - matrix: list of lists  (shape [n_summary_sents, n_gt_facts])
      - best_per_fact: best similarity for each GT fact (column-wise max)
      - gt_facts: list of GT fact strings  (column labels)
      - summary_sentences: list of summary sentence strings (row labels)
    """
    if isinstance(gt_facts, str):
        gt_facts = sent_tokenize(gt_facts)

    summary_sents = sent_tokenize(agent_summary) if isinstance(agent_summary, str) else list(agent_summary)
    summary_sents = [s.strip() for s in summary_sents if s.strip()]

    if not summary_sents or not gt_facts:
        return {
            "matrix": [],
            "best_per_fact": [],
            "gt_facts": gt_facts if isinstance(gt_facts, list) else [],
            "summary_sentences": summary_sents,
        }

    gt_embeds = _model.encode(gt_facts, convert_to_tensor=True, show_progress_bar=False)
    summary_embeds = _model.encode(summary_sents, convert_to_tensor=True, show_progress_bar=False)

    sim_matrix = util.cos_sim(summary_embeds, gt_embeds)  # [n_summary_sents, n_gt_facts]

    best_per_fact, _ = torch.max(sim_matrix, dim=0)

    return {
        "matrix": sim_matrix.cpu().tolist(),
        "best_per_fact": best_per_fact.cpu().tolist(),
        "gt_facts": gt_facts,
        "summary_sentences": summary_sents,
    }


import matplotlib.pyplot as plt
import textwrap
from pathlib import Path


def _truncate(text: str, max_chars: int = 80) -> str:
    """Truncate to max_chars on a word boundary, adding ellipsis if needed."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + " …"


def save_similarity_heatmap(sim_data: dict, save_path, agent_id: str = "", metric_label: str = "Cosine Similarity"):
    """
    Render the similarity matrix as an annotated heatmap and save to disk.

    Parameters
    ----------
    sim_data : dict returned by compute_similarity_matrix() or compute_bm25_matrix()
    save_path : str or Path — output PNG file path
    agent_id : str — used in the plot title
    metric_label : str — label for the colorbar and title (e.g. "Cosine Similarity", "BM25 Score")
    """
    matrix = sim_data.get("matrix", [])
    gt_facts = sim_data.get("gt_facts", [])
    summary_sents = sim_data.get("summary_sentences", [])

    if not matrix or not gt_facts or not summary_sents:
        return

    mat = np.array(matrix)
    n_rows, n_cols = mat.shape

    row_labels = [f"S{i+1}:  {_truncate(s, 72)}" for i, s in enumerate(summary_sents)]
    col_labels = [f"F{j+1}:  {_truncate(f, 72)}" for j, f in enumerate(gt_facts)]

    cell_h = max(0.7, min(1.2, 10.0 / n_rows))
    cell_w = max(1.4, min(2.2, 18.0 / n_cols))
    fig_w = max(10, n_cols * cell_w + 6)
    fig_h = max(5, n_rows * cell_h + 4)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "blue_red", ["#1e3a5f", "#3b7dd8", "#f0f0f0", "#e8644a", "#c0392b"]
    )
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    for i in range(n_rows):
        for j in range(n_cols):
            val = mat[i, j]
            color = "white" if val < 0.35 or val > 0.75 else "#1a1a2e"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=max(7, min(10, 120 // max(n_rows, n_cols))),
                    color=color, fontweight="bold")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=7.5, color="#c9d1d9")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=7.5, color="#c9d1d9")

    ax.set_xlabel("Ground Truth Facts", fontsize=11, color="#c9d1d9", labelpad=10)
    ax.set_ylabel("Agent Summary Sentences", fontsize=11, color="#c9d1d9", labelpad=10)

    title = f"{metric_label} Matrix — Agent {agent_id}" if agent_id else f"{metric_label} Matrix"
    ax.set_title(title, fontsize=14, color="#ffffff", fontweight="bold", pad=12)

    best = sim_data.get("best_per_fact", [])
    if best:
        subtitle = f"Best-per-fact avg: {np.mean(best):.3f}  |  Covered (>0.5): {sum(1 for b in best if b > 0.5)}/{len(best)}"
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
                fontsize=9, color="#8b949e", ha="center", va="bottom")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label(metric_label, fontsize=10, color="#c9d1d9")
    cbar.ax.tick_params(colors="#8b949e", labelsize=8)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor="#0d1117", edgecolor="none", bbox_inches="tight")
    plt.close(fig)

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


def compute_bm25_matrix(agent_summary: str, gt_facts) -> dict:
    """
    Compute the full BM25 score matrix between each summary sentence and
    each ground-truth fact, with sigmoid normalization into (0, 1).

    Returns a dict with:
      - matrix: list of lists  (shape [n_summary_sents, n_gt_facts])
      - best_per_fact: best normalized score per GT fact (column-wise max)
      - gt_facts: list of GT fact strings  (column labels)
      - summary_sentences: list of summary sentence strings (row labels)
    """
    if isinstance(gt_facts, str):
        gt_facts = sent_tokenize(gt_facts)

    summary_sents = sent_tokenize(agent_summary) if isinstance(agent_summary, str) else list(agent_summary)
    summary_sents = [s.strip() for s in summary_sents if s.strip()]

    if not summary_sents or not gt_facts:
        return {
            "matrix": [],
            "best_per_fact": [],
            "gt_facts": gt_facts if isinstance(gt_facts, list) else [],
            "summary_sentences": summary_sents,
        }

    token_sum = [word_tokenize(s.lower()) for s in summary_sents]
    token_gt = [word_tokenize(s.lower()) for s in gt_facts]

    bm25 = BM25Okapi(token_sum)

    # Build full matrix: each row is a summary sentence, each column is a GT fact.
    # BM25 naturally scores queries against documents, so we query each GT fact
    # against the summary-sentence corpus and transpose.
    raw_matrix = []
    for q in token_gt:
        raw_matrix.append(bm25.get_scores(q).tolist())
    # raw_matrix is [n_gt, n_summary] — transpose to [n_summary, n_gt]
    raw_matrix = list(map(list, zip(*raw_matrix)))

    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
    norm_matrix = [[float(sigmoid(v)) for v in row] for row in raw_matrix]

    best_per_fact = [max(norm_matrix[i][j] for i in range(len(summary_sents)))
                     for j in range(len(gt_facts))]

    return {
        "matrix": norm_matrix,
        "best_per_fact": best_per_fact,
        "gt_facts": gt_facts,
        "summary_sentences": summary_sents,
    }

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