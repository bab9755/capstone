from metrics import compute_final_score, compute_bert_score, compute_score
from runtime_config import get_ground_truth_bundle

agent_summary = """The event features busy, energetic career services. Counselors offer individual guidance. Workshops focus on career and interview skills. A career fair collects student resumes. Recruiters facilitate connections between students and employers."""
ground_truth_text = get_ground_truth_bundle().get("text", "")
score = compute_score(agent_summary, ground_truth_text)
print(score)