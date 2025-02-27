from sklearn.metrics import roc_auc_score, precision_score, recall_score
import numpy as np

def calculate_fdr_bound(W, q):
    W = np.array(W)
    W_abs = np.sort(np.abs(W))

    min_ratio = 99
    tau = -1
    for t in W_abs:
        ratio = (1 + np.sum(W <= -t)) / max(np.sum(W >= t), 1)
        if ratio <= q:
            return t
        if ratio < min_ratio:
            tau = t
            min_ratio = ratio

    print(f"Warning: No threshold can satisfy fdr<q, theshold: {tau}")
    return tau

def evalute_knockoff(results, labels, q):

    tau = calculate_fdr_bound(results, q)

    results = np.array(results)
    labels = np.array(labels)

    prediction = (results > tau).astype(int)

    auc = roc_auc_score(labels, results)
    precision = precision_score(labels, prediction)
    recall = recall_score(labels, prediction)

    return auc, precision, recall