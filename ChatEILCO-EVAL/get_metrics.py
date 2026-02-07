import json
import os
import requests
from collections import Counter
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# CONFIG
# ==========================
API_URL = "http://localhost:8000/chat"
SESSION_ID = "benchmark-session"
benchmark_folder = 'benchmarks'  
TEST_NUM = 1
DATASET_PATH = "dataset/final_norm.json"

# ==========================
# METRICS
# ==========================
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

models = {
    "paraphrase-multilingual-MiniLM-L12-v2": SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    ),
    "all-mpnet-base-v2": SentenceTransformer("all-mpnet-base-v2"),
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
    return rouge.score(gt, pred)["rougeL"].fmeasure


def semantic_similarity(pred, gt, model):
    emb_pred = model.encode(pred, convert_to_tensor=True)
    emb_gt = model.encode(gt, convert_to_tensor=True)
    return util.cos_sim(emb_pred, emb_gt).item()


# ==========================
# RETRIEVAL METRICS
# ==========================
def hit(retrieved_sources, golden_source):
    return int(golden_source in set(retrieved_sources))


def precision_at_k(retrieved, gold):
    if not retrieved:
        return 0.0
    return len(set(retrieved) & set(gold)) / len(retrieved)


def recall_at_k(retrieved, gold):
    if not gold:
        return 0.0
    return len(set(retrieved) & set(gold)) / len(set(gold))


# ==========================
# LOAD DATASET
# ==========================
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

results = []

# ==========================
# BENCHMARK LOOP
# ==========================
for i, e in enumerate(dataset):
    query = e["question"]
    gt = e["ground_truth"]

    gold_sources = [
        chunk["source"].split(".")[0]
        for chunk in e["source_chunks"]
    ]
    t = "Unknown"
    cl = e.get("source_chunks")

    if cl:
        context_type = cl[0].get("type")
        if context_type:
            t = context_type
    print(t)
    # ---- API CALL ----
    response = requests.post(
        API_URL,
        json={
            "session_id": f"{SESSION_ID}-{i}",
            "message": query,
        },
        timeout=60,
    )

    response.raise_for_status()
    data = response.json()

    pred = data["answer"]

    retrieved_sources = [
        s.get("source", "").split(".")[0]
        for s in data.get("sources", [])
    ]

    # ---- METRICS ----
    semantic_scores = {
        f"semantic_sim ({name})": semantic_similarity(pred, gt, model)
        for name, model in models.items()
    }

    results.append(
        {
            "question": query,
            "gt": gt,
            "pred": pred,
            "type":t,
            "hit": hit(retrieved_sources, gold_sources[0]),
            "f1 (generation)": f1_score(pred, gt),
            "rouge_l (generation)": rouge_l(pred, gt),
            "recall@k (retrieval)": recall_at_k(retrieved_sources, gold_sources),
            "precision@k (retrieval)": precision_at_k(
                retrieved_sources, gold_sources
            ),
            **semantic_scores,
        }
    )

# ==========================
# SAVE RESULTS
# ==========================
benchmark = pd.DataFrame(results)
os.makedirs(f"{benchmark_folder}", exist_ok=True)
benchmark.to_csv(f"{benchmark_folder}/bench_{TEST_NUM}.csv", index=False)
benchmark.to_excel(f"{benchmark_folder}/bench_{TEST_NUM}.xlsx", index=False)

# Définition des colonnes de métriques (Noms originaux dans l'Excel)
metric_cols = [
    "f1 (generation)",
    "rouge_l (generation)",
    "recall@k (retrieval)",
    "precision@k (retrieval)",
    "human_grading"
]

# Calcul de la similarité sémantique moyenne (si colonnes présentes)
numeric_cols = benchmark.select_dtypes(include=np.number).columns
sim_cols = [c for c in numeric_cols if c not in metric_cols and "id" not in c.lower()]
if sim_cols:
    benchmark["semantic_similarity_mean"] = benchmark[sim_cols].mean(axis=1)
    metric_cols.append("semantic_similarity_mean")

# --- TRADUCTION ---
# Dictionnaire de traduction pour l'affichage
translations = {
    "f1 (generation)": "F1 (Génération)",
    "rouge_l (generation)": "Rouge-L (Génération)",
    "recall@k (retrieval)": "Rappel@k (Recherche)",
    "precision@k (retrieval)": "Précision@k (Recherche)",
    "human_grading": "Évaluation Humaine",
    "semantic_similarity_mean": "Similitude Sémantique",
    "text": "Texte",
    "table": "Tableau"
}

# Appliquer la traduction aux types de données dans le DataFrame
benchmark["type"] = benchmark["type"].map(translations).fillna(benchmark["type"])

# Liste des métriques en français pour l'axe X
french_metrics = [translations.get(m, m) for m in metric_cols]

# ---------------------------------------------------------
# 2. Calcul des Statistiques
# ---------------------------------------------------------
# A. Moyenne par Type (Texte vs Tableau)
grouped_means = benchmark.groupby("type")[metric_cols].mean()

# B. Moyenne Globale (Pondérée sur tout le dataset)
global_means = benchmark[metric_cols].mean()

# Afficher le rapport textuel
print("--- Rapport Statistique (Moyennes) ---")
report = grouped_means.T
report.index = french_metrics # Mettre les index en français
report["Moyenne Globale"] = global_means.values
print(report.round(3))

# ---------------------------------------------------------
# 3. Graphique 1 : Performance Globale (Moyenne)
# ---------------------------------------------------------
plt.figure(figsize=(12, 7))
ax1 = plt.gca()

# Création des barres
bars = ax1.bar(french_metrics, global_means.values, color='#2ca02c', alpha=0.8, width=0.6)

# Titres et Labels
plt.title(f"Performance Moyenne Globale (n={len(benchmark)})", fontsize=14, pad=15)
plt.ylabel("Score Moyen (0-1)", fontsize=12)
plt.ylim(0, 1.15) # Marge pour les étiquettes
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.xticks(rotation=30, ha='right', fontsize=10)

# Ajouter les valeurs sur les barres
ax1.bar_label(bars, fmt='%.2f', padding=3, fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('benchmarks/global_average_french.png', dpi=300)
plt.show()

# ---------------------------------------------------------
# 4. Graphique 2 : Comparaison par Type (Texte vs Tableau)
# ---------------------------------------------------------
plt.figure(figsize=(14, 8))
ax2 = plt.gca()

x = np.arange(len(french_metrics))
width = 0.35

# Récupérer les types (maintenant traduits en français : Texte, Tableau)
types = grouped_means.index.tolist()
colors = ['#ff7f0e', '#1f77b4', '#d62728'] # Orange, Bleu, Rouge

for i, t in enumerate(types):
    offset = (i - len(types)/2) * width + width/2
    values = grouped_means.loc[t].values
    
    # Création des barres groupées
    bars = ax2.bar(x + offset, values, width, label=t, color=colors[i % len(colors)], alpha=0.9)
    
    # Ajouter les valeurs sur CHAQUE barre
    ax2.bar_label(bars, fmt='%.2f', padding=3, fontsize=10)

# Titres et Labels
plt.title("Comparaison des Performances : Texte vs Tableau", fontsize=16, pad=20)
plt.ylabel("Score Moyen (0-1)", fontsize=12)
plt.ylim(0, 1.15)
plt.xticks(x, french_metrics, rotation=30, ha='right', fontsize=10)
plt.legend(title="Type de Donnée", fontsize=12, loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('{benchmark_folder}/type_comparison_french.png', dpi=300)
plt.show()