import json
import ast

def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    

def compute_recall_at_k(results, ks=(1,5,10,20)):
    total = len(results)
    hit_counts = {k: 0 for k in ks}
    for qid, entry in results.items():
        gt_list = entry["ground_truth"]
        gt_set = set(gt_list)

        retrieved = [x["id"] for x in entry["retrieved_entities"]]

        for k in ks:
            topk = retrieved[:k]
            if any(r in gt_set for r in topk):
                hit_counts[k] += 1



    recall = {k: hit_counts[k] / total for k in ks}
    return recall


# ----------------- usage -----------------
path = "results/mbeir_results/oven_ours.json"
results = load_results(path)

recall = compute_recall_at_k(results)
print(recall)
