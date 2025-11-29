import os
import sys
import random
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

import yaml

import warnings

from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=FutureWarning)


def compute_det_curve(target_scores, nontarget_scores):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))) # 1 is bonafide and 0 is spoof

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - \
        (np.arange(1, n_scores + 1) - tar_trial_sums)

    # false rejection rates
    frr = np.concatenate(
        (np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums /
                          nontarget_scores.size))  # false acceptance rates
    # Thresholds are the sorted scores
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


def compute_fpr_at_tpr95(target_scores, nontarget_scores, desired_tpr=0.95):
    """
    FPR at 95% TPR (linear interpolation).
    Conventions:
      - target_scores: positives (here = OOD/bonafide), higher = target
      - nontarget_scores: negatives (here = ID/spoof)
    """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    tpr = 1.0 - frr  # TPR for target (OOD)

    # Make TPR increasing for np.interp
    tpr_inc = tpr[::-1]
    far_inc = far[::-1]
    thr_inc = thresholds[::-1]

    # Clamp query into achievable range
    tpr_query = np.clip(desired_tpr, tpr_inc[0], tpr_inc[-1])

    fpr95 = np.interp(tpr_query, tpr_inc, far_inc)
    thr95 = np.interp(tpr_query, tpr_inc, thr_inc)
    return fpr95, thr95


def compute_auroc_from_det(target_scores: np.ndarray, nontarget_scores: np.ndarray) -> float:
    frr, far, _ = compute_det_curve(target_scores, nontarget_scores)
    fpr = far
    tpr = 1.0 - frr

    fpr_inc = fpr[::-1]
    tpr_inc = tpr[::-1]

    auroc = np.trapz(tpr_inc, fpr_inc)
    return float(auroc)



def align_scores_for_target(target_scores, nontarget_scores):
    t = np.asarray(target_scores, dtype=float).ravel()
    n = np.asarray(nontarget_scores, dtype=float).ravel()
    flipped = False
    if np.median(t) <= np.median(n):
        t, n = -t, -n
        flipped = True
    return t, n, flipped


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norm, eps, None)


def deep_knn_scores(
    train_embeds_path: str,
    id_embeds_path: str,
    ood_embeds_path: str,
    k: int = 10,
    n_jobs: int = -1,
    ):
    
    train_embeds = np.load(train_embeds_path)
    id_embeds    = np.load(id_embeds_path)
    ood_embeds   = np.load(ood_embeds_path)
    
    Xtr = _l2_normalize(train_embeds.astype(np.float32, copy=False))
    Xid = _l2_normalize(id_embeds.astype(np.float32, copy=False))
    Xood = _l2_normalize(ood_embeds.astype(np.float32, copy=False))

    k_eff = min(k, len(Xtr))
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean", n_jobs=n_jobs)
    nn.fit(Xtr)

    dist_id,  _ = nn.kneighbors(Xid,  n_neighbors=k_eff, return_distance=True)
    dist_ood, _ = nn.kneighbors(Xood, n_neighbors=k_eff, return_distance=True)
    
    print("ID distance shape:", dist_id.shape)
    print("OOD distance shape:", dist_ood.shape)

    id_scores  = dist_id[:,  k_eff - 1]
    ood_scores = dist_ood[:, k_eff - 1] 
    
    ood_scores, id_scores, flipped = align_scores_for_target(ood_scores, id_scores)
    
    eer_score, eer_threshold = compute_eer(ood_scores, id_scores)
    print(f"EER: {eer_score*100:.2f}% | flipped={flipped}")

    fpr95, thr95 = compute_fpr_at_tpr95(ood_scores, id_scores, desired_tpr=0.95)
    print(f"FPR@95TPR (OOD=bonafide) = {fpr95*100:.2f}%  |  threshold = {thr95:.6f} | flipped={flipped}")
    
    auroc = compute_auroc_from_det(ood_scores, id_scores)
    print(f"AUROC: {auroc * 100:.2f}% | flipped={flipped}")
    
    plt.figure(figsize=(7,5))
    # KDE curves
    sns.kdeplot(ood_scores, bw_adjust=0.5, fill=True,
                color='tab:blue', alpha=0.4, label='Bonafide (target/OOD)')
    sns.kdeplot(id_scores, bw_adjust=0.5, fill=True,
                color='tab:orange', alpha=0.4, label='Spoof (nontarget/ID)')

    plt.title("KNN-based score distribution of OOD and ID samples")
    plt.xlabel("KNN-based score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("ID_OOD_distribution.png", dpi=200)
    plt.show()


def knn_indegree_scores(
    train_embeds_path: str,
    id_embeds_path: str,
    ood_embeds_path: str,
    k: int = 10,          
    M: int = 100,          
    metric: str = "euclidean",
    l2_normalize: bool = True,
    n_jobs: int = -1,
    ):
    
    assert metric in ("euclidean", "cosine")
    
    train_embeds = np.load(train_embeds_path)
    ID_embeds    = np.load(id_embeds_path)
    OOD_embeds   = np.load(ood_embeds_path)

    Xtr = _l2_normalize(train_embeds) if l2_normalize else np.asarray(train_embeds, np.float32)
    Xid = _l2_normalize(ID_embeds) if l2_normalize else np.asarray(ID_embeds, np.float32)
    Xood = _l2_normalize(OOD_embeds) if l2_normalize else np.asarray(OOD_embeds, np.float32)

    n_train = len(Xtr)

    k_eff = min(k, n_train - 1)
    nn_tr = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric, n_jobs=n_jobs).fit(Xtr)
    dist_tr, idx_tr = nn_tr.kneighbors(Xtr, return_distance=True)
    dist_tr = dist_tr[:, 1:]   # shape [Ntr, k_eff]
    kth_dist_train = dist_tr[:, k_eff - 1]  

    def _score_block(Xtest: np.ndarray):
        M_eff = min(M, n_train)
        nn_test = NearestNeighbors(n_neighbors=M_eff, metric=metric, n_jobs=n_jobs).fit(Xtr)
        dists, idxs = nn_test.kneighbors(Xtest, return_distance=True)   # [N, M_eff], [N, M_eff]

        kth_thresh = kth_dist_train[idxs]                                # [N, M_eff]

        indeg = (dists <= kth_thresh).sum(axis=1).astype(np.int32)       # [N]

        scores = 1.0 - (indeg.astype(np.float32) / float(M_eff))
        return scores, indeg, M_eff

    id_scores,  id_indeg,  _ = _score_block(Xid)
    ood_scores, ood_indeg, _ = _score_block(Xood)

    ood_scores, id_scores, flipped = align_scores_for_target(ood_scores, id_scores)
    
    eer_score, eer_threshold = compute_eer(ood_scores, id_scores)
    print(f"EER: {eer_score*100:.2f}% | flipped={flipped}")

    fpr95, thr95 = compute_fpr_at_tpr95(ood_scores, id_scores, desired_tpr=0.95)
    print(f"FPR@95TPR (OOD=bonafide) = {fpr95*100:.2f}%  |  threshold = {thr95:.6f} | flipped={flipped}")
    
    auroc = compute_auroc_from_det(ood_scores, id_scores)
    print(f"AUROC: {auroc * 100:.2f}% | flipped={flipped}")
    
    plt.figure(figsize=(7,5))
    # KDE curves
    sns.kdeplot(ood_scores, bw_adjust=0.5, fill=True,
                color='tab:blue', alpha=0.4, label='Bonafide (target/OOD)')
    sns.kdeplot(id_scores, bw_adjust=0.5, fill=True,
                color='tab:orange', alpha=0.4, label='Spoof (nontarget/ID)')

    # plt.yscale('log')
    plt.title("KNNindegree-based score distribution of OOD and ID samples")
    plt.xlabel("KNNindegree-based score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("ID_OOD_distribution.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    
    print("Starting evaluation of deepfake detection...")
    

    exp_name = 'ST_RB_SSLAASIST_MTL_KANaux_trainingScratch'
    
    embds_folder = os.path.join('./extracted_embds', exp_name)
    assert os.path.exists(embds_folder), f"Folder {embds_folder} does not exist."
    
    
    ####
    #### AttackAttr embeddings (50-d features)
    ####
    ASVspoof2019_attr17_train_attackAttr_embds_path = os.path.join(embds_folder, 'ASVspoof2019_attr17_train_attackAttr_embeddings.npy')
    # ASVspoof2019_attr17_dev_attackAttr_embds_path = os.path.join(embds_folder, 'ASVspoof2019_attr17_dev_attackAttr_embeddings.npy')
    ASVspoof2019_attr17_eval_attackAttr_embds_path = os.path.join(embds_folder, 'ASVspoof2019_attr17_eval_attackAttr_embeddings.npy')
    ASVspoof2019_attr2_full_bonafide_attackAttr_embds_path = os.path.join(embds_folder, 'ASVspoof2019_attr2_full_bonafide_attackAttr_embeddings.npy')
    
    ####
    #### AttackCls embeddings (17-d features)
    ####
    ASVspoof2019_attr17_train_attackCls_embds_path = os.path.join(embds_folder, 'ASVspoof2019_attr17_train_attackCls_embeddings.npy')
    # ASVspoof2019_attr17_dev_attackCls_embds_path = os.path.join(embds_folder, 'ASVspoof2019_attr17_dev_attackCls_embeddings.npy')
    ASVspoof2019_attr17_eval_attackCls_embds_path = os.path.join(embds_folder, 'ASVspoof2019_attr17_eval_attackCls_embeddings.npy')
    ASVspoof2019_attr2_full_bonafide_attackCls_embds_path = os.path.join(embds_folder, 'ASVspoof2019_attr2_full_bonafide_attackCls_embeddings.npy')
    
    
    # OOD detection using Deep-KNN (Distance-based algorithm)
    deep_knn_scores(
        train_embeds_path=ASVspoof2019_attr17_train_attackAttr_embds_path,
        id_embeds_path=ASVspoof2019_attr17_eval_attackAttr_embds_path,
        ood_embeds_path=ASVspoof2019_attr2_full_bonafide_attackAttr_embds_path,
        k=10)
    
    # OOD detection using KNN-indegree (Indegree-based algorithm)
    knn_indegree_scores(
        train_embeds_path=ASVspoof2019_attr17_train_attackAttr_embds_path,
        id_embeds_path=ASVspoof2019_attr17_eval_attackAttr_embds_path,
        ood_embeds_path=ASVspoof2019_attr2_full_bonafide_attackAttr_embds_path,
        k=2000,
        M=2000)
    