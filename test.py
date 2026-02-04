import json,os
from collections import Counter
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import pandas as pd,matplotlib.pyplot as plt

from RAG import RAG, Context
from Knowledge_base import LLamaCppEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


# Initialize ROUGE scorer and French-optimized embedding model
transformer = 'paraphrase-multilingual-MiniLM-L12-v2'
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
models = { 'paraphrase-multilingual-MiniLM-L12-v2':SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2"),
          "all-mpnet-base-v2":SentenceTransformer("all-mpnet-base-v2")
          }


def f1_score(pred, gt):
    pred_tokens = pred.split()
    gt_tokens = gt.split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * (precision * recall) / (precision + recall)

def rouge_l(pred, gt):
    score = rouge.score(gt, pred)
    return score['rougeL'].fmeasure

def semantic_similarity(pred, gt, model: SentenceTransformer):
    emb_pred = model.encode(pred, convert_to_tensor=True)
    emb_gt = model.encode(gt, convert_to_tensor=True)
    return util.cos_sim(emb_pred, emb_gt).item()


def extract_sources(docs, k=None):
    """
    Extrait les noms de fichiers sources (dédupliqués)
    depuis une liste de documents/chunks.

    docs: list of documents with metadata['source']
    k: int or None (top-k)
    """
    if k is not None:
        docs = docs[:k]

    a = {
        os.path.split(doc.metadata["source"])[-1].split(".")[0]
        for doc in docs
        if hasattr(doc, "metadata") and "source" in doc.metadata
    }
    return a

def hit(retrieved_sources: list, golden_source:str):
    return int(golden_source in set(retrieved_sources))

def _relevant_retrieved(retrieved_docs, gold_sources, k=None):
    """
    Retourne le nombre de documents pertinents récupérés parmi top-k
    """
    # transformer en set
    gold_sources_set = set(gold_sources)

    if k is None:
        retrieved_docs_top = retrieved_docs
        k = len(retrieved_docs_top)
    else:
        retrieved_docs_top = retrieved_docs[:k]

    retrieved_docs_set = set(retrieved_docs_top)

    return len(retrieved_docs_set & gold_sources_set)


def precision_at_k(retrieved_docs: list[str], gold_sources: list[str], k=None):
    relevant_retrieved = _relevant_retrieved(retrieved_docs, gold_sources, k)
    
    if k is None:
        k = len(retrieved_docs)  
    return relevant_retrieved / k if k > 0 else 0.0


def recall_at_k(retrieved_docs: list[str], gold_sources: list[str], k=None):
    relevant_retrieved = _relevant_retrieved(retrieved_docs, gold_sources, k)
    gold_sources_set = set(gold_sources)
    return relevant_retrieved / len(gold_sources_set) if gold_sources_set else 0.0



# for entry in dataset:
#     pred = entry.get("answer", "")
#     gt = entry.get("ground_truth", "")
    
#     em = exact_match(pred, gt)
#     f1 = f1_score(pred, gt)
#     rouge_l_score = rouge_l(pred, gt)
#     sem_sim = semantic_similarity(pred, gt)
    
#     results.append({
#         "id": entry.get("id"),
#         "EM": em,
#         "F1": f1,
#         "ROUGE-L": rouge_l_score,
#         "Semantic_Similarity": sem_sim
#     })

API_URL = "http://localhost:8000/v1/chat/completions"
EMB_URL = "http://localhost:8080/v1/embeddings"
EMB_MODEL_NAME = "jina"
MODEL_NAME = "mistral"

emb = LLamaCppEmbeddings(EMB_MODEL_NAME, EMB_URL)
emb2 = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
EMB = emb2
VECTOR_PATH = "./faiss_index-v7"
test_num = 12

rag = RAG(API_URL,MODEL_NAME)

ctx = Context(0,0)

rag.load_knowledge_base(EMB, VECTOR_PATH)
with open("final.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

results = []

for e in dataset:
    
    query = e['question']
    gt = e['ground_truth']
    gold_sources = [chunk["source_file"].split(".")[0] for chunk in e['source_chunks']]

    #rewritten_query = rag.rewrite_query(query, ctx)
    docs = rag.retriever(query)
    retrieved_docs = extract_sources(docs)
    sys_prompt = rag.get_system_prompt(docs)
    user_entry = {"role": "user", "content": query}
    messages = [{"role": "system", "content": sys_prompt}] + [user_entry]

    response = rag.chat_completion(messages, {"temperature": 0.1, "stream": False,"model":"mistral"})
    pred = response['choices'][0]["message"]["content"]
    st={f"semantic_sim ({model})": semantic_similarity(pred,gt, item) for model,item in models.items()}
    results.append(
        {"question": query,
        "gt":gt,
        "pred":pred,
        "hit": hit(retrieved_docs,gold_sources[0]),
        "f1 (generation)":f1_score(pred,gt),
        "rouge_l (generation)":rouge_l(pred, gt),
        "recall@k (retrieval)":recall_at_k(retrieved_docs, gold_sources),
        "precision@k (retrieval)":precision_at_k(retrieved_docs, gold_sources),
        **st
        }
    )

df = pd.DataFrame(results)
df.to_csv(f"benchmarks/bench_{test_num}.csv")


metric_cols = [
    "f1 (generation)",
    "rouge_l (generation)",
    "recall@k (retrieval)",
    "precision@k (retrieval)",
    "hit"
]

sim_cols = [col for col in df.columns if col not in metric_cols and col not in ["question", "gt", "pred", "hit"]]

df["semantic_similarity_mean"] = df[sim_cols].mean(axis=1)

all_metrics = metric_cols + ["semantic_similarity_mean"]

means = df[metric_cols].mean()

mean_scores = df[all_metrics].mean()

plt.figure(figsize=(8,5))
plt.bar(mean_scores.index, mean_scores.values, color="skyblue")
plt.title("Mean Scores Across Metrics")
plt.ylabel("Mean Score")
plt.ylim(0,1)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(f"benchmarks/metrics_mean_bar_{test_num}.png")
plt.close()






