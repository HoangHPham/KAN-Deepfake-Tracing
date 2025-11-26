import os
import sys
import random
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

import yaml

import warnings

from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=FutureWarning)


def sme_energy(logits, T=1.0):
    """
    Compute Softmax-based Energy (SME) score for OOD detection.

    Args:
        logits: Tensor [B, K] - raw model outputs
        T: float - temperature

    Returns:
        energy: Tensor [B] - SME energy scores
    """
    ### Softmax energy
    # logits_T = logits / T
    # probs = F.softmax(logits_T, dim=1)  # [B, K]
    # energy = -T * torch.log(torch.sum(torch.exp(probs), dim=1))
    
    ### Traditional energy
    energy = -T * torch.logsumexp(logits / T, dim=1)

    # MSP
    # energy = torch.softmax(logits / T, dim=1).max(dim=1)

    # entropy
    # p = torch.softmax(logits / T, dim=1)
    # energy = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=1)
    
    return energy.detach().cpu().numpy()
    # return energy.values.detach().cpu().numpy()


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
    """
    Tính AUROC dùng compute_det_curve.
    Quy ước: target_scores = lớp dương (higher = target), nontarget_scores = lớp âm.
    """
    frr, far, _ = compute_det_curve(target_scores, nontarget_scores)
    # ROC: FPR = far, TPR = 1 - frr.
    fpr = far
    tpr = 1.0 - frr

    # Đảm bảo FPR tăng dần để tính diện tích chuẩn (trapezoid)
    fpr_inc = fpr[::-1]
    tpr_inc = tpr[::-1]

    # Tính AUROC (diện tích dưới đường TPR(FPR))
    auroc = np.trapz(tpr_inc, fpr_inc)
    return float(auroc)


def evaluate_df_detection(ID_embeddings_path=None, OOD_embeddings_path=None):
    
    # get logits
    ID_embds = np.load(ID_embeddings_path)
    OOD_embds = np.load(OOD_embeddings_path)
    
    # Compute SME energy
    ID_energy_scores = sme_energy(torch.tensor(ID_embds, dtype=torch.float32), T=1) # lower energy = more likely to be ID (0)
    OOD_energy_scores = sme_energy(torch.tensor(OOD_embds, dtype=torch.float32), T=1) # higher energy = more likely to be OOD (1)
    
    eer_score, eer_threshold = compute_eer(OOD_energy_scores, ID_energy_scores)
    print(f"EER: {eer_score*100:.2f}%")

    fpr95, thr95 = compute_fpr_at_tpr95(OOD_energy_scores, ID_energy_scores, desired_tpr=0.95)
    print(f"FPR@95TPR (OOD=bonafide) = {fpr95*100:.2f}%  |  threshold = {thr95:.6f}")
    
    auroc = compute_auroc_from_det(OOD_energy_scores, ID_energy_scores)
    print("AUROC: {:.2f}%".format(auroc * 100))
    
    plt.figure(figsize=(7,5))
    # KDE curves
    sns.kdeplot(OOD_energy_scores, bw_adjust=0.5, fill=True,
                color='tab:blue', alpha=0.4, label='Bonafide (target/OOD)')
    sns.kdeplot(ID_energy_scores, bw_adjust=0.5, fill=True,
                color='tab:orange', alpha=0.4, label='Spoof (nontarget/ID)')

    plt.title("Energy distribution of OOD and ID samples")
    plt.xlabel("Energy score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("ID_OOD_distribution.png", dpi=200)
    plt.show()


def align_scores_for_target(target_scores, nontarget_scores):
    """
    Đảm bảo quy ước: higher = target.
    Nếu median(target) <= median(nontarget) thì đảo dấu cả hai.
    """
    t = np.asarray(target_scores, dtype=float).ravel()
    n = np.asarray(nontarget_scores, dtype=float).ravel()
    flipped = False
    if np.median(t) <= np.median(n):
        t, n = -t, -n
        flipped = True
    return t, n, flipped


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Chuẩn hoá L2 theo hàng: X[i] /= ||X[i]||_2."""
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norm, eps, None)


def deep_knn_scores(
    train_embeds_path: str,
    id_embeds_path: str,
    ood_embeds_path: str,
    k: int = 10,
    n_jobs: int = -1,
    ):
    """
    Tính điểm KNN cho ID và OOD theo thuật toán Deep Nearest Neighbors.
    - Chuẩn hoá L2 toàn bộ features.
    - Với mỗi mẫu test, score = khoảng cách tới láng giềng thứ k gần nhất trong train.
    
    Args:
        train_embeds: np.ndarray [N_train, D]
        ID_embeds   : np.ndarray [N_id, D]   (in-distribution: 17 attacks)
        OOD_embeds  : np.ndarray [N_ood, D]  (out-of-distribution: bonafide)
        k           : số láng giềng (1-indexed trong paper; mặc định 10)
        n_jobs      : số luồng cho NearestNeighbors (-1 = dùng tất cả)
    Returns:
        id_scores   : np.ndarray [N_id]   (lớn hơn ⇒ OOD-like hơn)
        ood_scores  : np.ndarray [N_ood]  (lớn hơn ⇒ OOD-like hơn)
    """
    
    train_embeds = np.load(train_embeds_path)
    id_embeds    = np.load(id_embeds_path)
    ood_embeds   = np.load(ood_embeds_path)
    
    # 1) L2-normalize
    Xtr = _l2_normalize(train_embeds.astype(np.float32, copy=False))
    Xid = _l2_normalize(id_embeds.astype(np.float32, copy=False))
    Xood = _l2_normalize(ood_embeds.astype(np.float32, copy=False))

    # 2) KNN index (Euclidean trên vector đã norm ~ cosine distance)
    k_eff = min(k, len(Xtr))
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean", n_jobs=n_jobs)
    nn.fit(Xtr)

    # 3) Lấy khoảng cách tới k láng giềng gần nhất (đã sắp tăng dần)
    dist_id,  _ = nn.kneighbors(Xid,  n_neighbors=k_eff, return_distance=True)
    dist_ood, _ = nn.kneighbors(Xood, n_neighbors=k_eff, return_distance=True)
    
    print("ID distance shape:", dist_id.shape)
    print("OOD distance shape:", dist_ood.shape)

    # 4) Score = khoảng cách tới láng giềng thứ k (theo paper)
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
    k: int = 10,           # k của đồ thị kNN trên train
    M: int = 100,          # số ứng viên reverse-kNN khi đánh giá test (M >= k)
    metric: str = "euclidean",
    l2_normalize: bool = True,
    n_jobs: int = -1,
    ):
    """
    Tính score theo kNN–indegree:
      - Xây đồ thị kNN (hướng) trên tập train để lấy ngưỡng d_k(i) (k-th NN distance của từng điểm train).
      - Với mỗi mẫu test x: tìm M láng giềng train gần nhất; đếm có bao nhiêu điểm train i
        thoả d(x, z_i) <= d_k(i). Đó là indegree ước lượng của x.
      - Trả về score = 1 - indegree/M  (∈ [0,1]), nên *cao ⇒ OOD-like*.

    Returns:
      id_scores, ood_scores  (đã chuẩn hoá để cao ⇒ OOD)
    """
    assert metric in ("euclidean", "cosine")
    
    train_embeds = np.load(train_embeds_path)
    ID_embeds    = np.load(id_embeds_path)
    OOD_embeds   = np.load(ood_embeds_path)

    # 0) Chuẩn hoá (nên bật để Euclid ~ cosine)
    Xtr = _l2_normalize(train_embeds) if l2_normalize else np.asarray(train_embeds, np.float32)
    Xid = _l2_normalize(ID_embeds) if l2_normalize else np.asarray(ID_embeds, np.float32)
    Xood = _l2_normalize(OOD_embeds) if l2_normalize else np.asarray(OOD_embeds, np.float32)

    n_train = len(Xtr)

    # 1) Xây kNN trên train để lấy d_k(i)
    k_eff = min(k, n_train - 1)
    nn_tr = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric, n_jobs=n_jobs).fit(Xtr)
    # self + k neighbors → bỏ cột đầu (self)
    dist_tr, idx_tr = nn_tr.kneighbors(Xtr, return_distance=True)
    dist_tr = dist_tr[:, 1:]   # shape [Ntr, k_eff]
    kth_dist_train = dist_tr[:, k_eff - 1]   # d_k(i) cho từng điểm train, shape [Ntr]

    # 2) Hàm tiện ích: tính indegree & score cho một tập test
    def _score_block(Xtest: np.ndarray):
        M_eff = min(M, n_train)
        # Lấy M ứng viên train gần nhất của từng test
        nn_test = NearestNeighbors(n_neighbors=M_eff, metric=metric, n_jobs=n_jobs).fit(Xtr)
        dists, idxs = nn_test.kneighbors(Xtest, return_distance=True)   # [N, M_eff], [N, M_eff]

        # Ngưỡng d_k(i) tương ứng từng hàng (test) – từng cột (neighbor)
        kth_thresh = kth_dist_train[idxs]                                # [N, M_eff]

        # Đếm xem test có lọt vào top-k của bao nhiêu điểm train (reverse-kNN)
        indeg = (dists <= kth_thresh).sum(axis=1).astype(np.int32)       # [N]

        # Score cao ⇒ OOD: 1 - (indegree / M_eff)
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


def knn_weighted_indegree_scores(
    train_embeds_path: str,
    id_embeds_path: str,
    ood_embeds_path: str,
    k: int = 10,            # k của đồ thị kNN trên train
    M: int = 100,           # số ứng viên reverse-kNN khi đánh giá test (M >= k)
    metric: str = "euclidean",
    l2_normalize: bool = True,
    n_jobs: int = -1,
    sigma: float | None = None,   # None => dùng sigma cục bộ = d_k(i); hoặc truyền số float cho sigma toàn cục
    sigma_clip: float = 1e-6      # tránh chia 0 khi sigma rất nhỏ
):
    """
    Tính score theo kNN–Weighted indegree (Gaussian kernel):

      - Xây kNN trên train để lấy d_k(i) (k-th NN distance cho từng điểm train).
      - Với mỗi mẫu test x: tìm M láng giềng train gần nhất.
        * Với từng neighbor (i):
            + Nếu d(x, z_i) <= d_k(i): đóng góp trọng số w = exp(- d^2 / (2 * sigma_i^2))
              (mặc định sigma_i = d_k(i); nếu 'sigma' là số thì dùng sigma_i = sigma).
            + Nếu d(x, z_i) > d_k(i): w = 0 (không đóng góp).
        * Weighted indegree = tổng các w (tối đa M).
      - Trả về score = 1 - (weighted_indegree / M)  ∈ [0, 1], nên **cao ⇒ OOD-like**.

    Returns:
      id_scores, ood_scores
    """
    assert metric in ("euclidean", "cosine")

    # Load
    train_embeds = np.load(train_embeds_path)
    ID_embeds    = np.load(id_embeds_path)
    OOD_embeds   = np.load(ood_embeds_path)

    # (0) Chuẩn hoá
    Xtr = _l2_normalize(train_embeds) if l2_normalize else np.asarray(train_embeds, np.float32)
    Xid = _l2_normalize(ID_embeds)    if l2_normalize else np.asarray(ID_embeds,    np.float32)
    Xood = _l2_normalize(OOD_embeds)  if l2_normalize else np.asarray(OOD_embeds,  np.float32)

    n_train = len(Xtr)
    if n_train < 2:
        raise ValueError("train_embeds phải có ít nhất 2 mẫu.")

    # (1) kNN trên train để lấy d_k(i)
    k_eff = min(k, n_train - 1)
    nn_tr = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric, n_jobs=n_jobs).fit(Xtr)
    dist_tr, _ = nn_tr.kneighbors(Xtr, return_distance=True)  # self + k neighbors
    dist_tr = dist_tr[:, 1:]                                  # bỏ self
    kth_dist_train = dist_tr[:, k_eff - 1].astype(np.float32) # [Ntr]

    # (2) Hàm tính weighted indegree cho một tập test
    def _weighted_scores(Xtest: np.ndarray):
        M_eff = min(M, n_train)
        nn_test = NearestNeighbors(n_neighbors=M_eff, metric=metric, n_jobs=n_jobs).fit(Xtr)
        dists, idxs = nn_test.kneighbors(Xtest, return_distance=True)  # [N, M_eff]

        # Ngưỡng d_k(i) và sigma_i tương ứng cho từng neighbor
        kth_thresh = kth_dist_train[idxs]                               # [N, M_eff]
        if sigma is None:
            sigma_mat = np.clip(kth_thresh, sigma_clip, None)           # sigma_i = d_k(i)
        else:
            sigma_mat = np.full_like(kth_thresh, float(sigma))

        # mask trong-biên: chỉ những neighbor mà test sẽ lọt top-k của i
        mask = (dists <= kth_thresh)

        # Gaussian weight: w = exp( -d^2 / (2*sigma_i^2) ), ngoài biên → 0
        # để ổn định số học:
        denom = 2.0 * (sigma_mat ** 2)
        denom = np.clip(denom, sigma_clip**2, None)
        w = np.exp(-(dists ** 2) / denom) * mask

        indeg_w = w.sum(axis=1)                        # tổng trọng số ≤ M_eff
        scores = 1.0 - (indeg_w / float(M_eff))        # cao ⇒ OOD-like, ∈ [0,1]
        return scores

    id_scores  = _weighted_scores(Xid)
    ood_scores = _weighted_scores(Xood)

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
    
    ### deepfake detection evaluation
    # ASVspoof2019_attr17_eval_attackCls_embds_path = os.path.join(save_folder, 'ASVspoof2019_attr17_eval_attackCls_embeddings_AASIST_MTL_KANaux_fullFinetuning.npy')
    # ASVspoof2019_attr2_full_bonafide_attackCls_embds_path = os.path.join(save_folder, 'ASVspoof2019_attr2_full_bonafide_attackCls_embeddings_AASIST_MTL_KANaux_fullFinetuning.npy')
    
    # evaluate_df_detection(ID_embeddings_path=ASVspoof2019_attr17_eval_attackCls_embds_path,
    #                         OOD_embeddings_path=ASVspoof2019_attr2_full_bonafide_attackCls_embds_path)

    ### OOD detection using KNN
    # attackAttr_embds_folder = './AttackAttr_embds'
    # ASVspoof2019_attr17_train_attackAttr_embds_path = os.path.join(attackAttr_embds_folder, 'ASVspoof2019_attr17_train_attackAttr_embeddings_AASIST_MTL_KANaux_fullFinetuning.npy')
    # ASVspoof2019_attr17_eval_attackAttr_embds_path = os.path.join(attackAttr_embds_folder, 'ASVspoof2019_attr17_eval_attackAttr_embeddings_AASIST_MTL_KANaux_fullFinetuning.npy')
    # ASVspoof2019_attr2_full_bonafide_attackAttr_embds_path = os.path.join(attackAttr_embds_folder, 'ASVspoof2019_attr2_full_bonafide_attackAttr_embeddings_AASIST_MTL_KANaux_fullFinetuning.npy')
    
    
    exp_name = 'ST_RB_SSLAASIST_MTL_KANaux_fullFinetuning'
    
    embds_folder = os.path.join('./extracted_embds', exp_name)
    assert os.path.exists(embds_folder), f"Folder {embds_folder} does not exist."
    
    
    ####
    #### AttackAttr embeddings (50-d features)
    ####
    # ASVspoof2019_attr17_train_attackAttr_embds_path = os.path.join(embds_folder, 'ASVspoof2019_attr17_train_attackAttr_embeddings.npy')
    # # ASVspoof2019_attr17_dev_attackAttr_embds_path = os.path.join(embds_folder, 'ASVspoof2019_attr17_dev_attackAttr_embeddings.npy')
    # ASVspoof2019_attr17_eval_attackAttr_embds_path = os.path.join(embds_folder, 'ASVspoof2019_attr17_eval_attackAttr_embeddings.npy')
    # ASVspoof2019_attr2_full_bonafide_attackAttr_embds_path = os.path.join(embds_folder, 'ASVspoof2019_attr2_full_bonafide_attackAttr_embeddings.npy')
    
    ####
    #### AttackCls embeddings (17-d features)
    ####
    ASVspoof2019_attr17_train_attackCls_embds_path = os.path.join(embds_folder, 'ASVspoof2019_attr17_train_attackCls_embeddings.npy')
    # ASVspoof2019_attr17_dev_attackCls_embds_path = os.path.join(embds_folder, 'ASVspoof2019_attr17_dev_attackCls_embeddings.npy')
    ASVspoof2019_attr17_eval_attackCls_embds_path = os.path.join(embds_folder, 'ASVspoof2019_attr17_eval_attackCls_embeddings.npy')
    ASVspoof2019_attr2_full_bonafide_attackCls_embds_path = os.path.join(embds_folder, 'ASVspoof2019_attr2_full_bonafide_attackCls_embeddings.npy')
    
    
    # deep_knn_scores(
    #     train_embeds_path=ASVspoof2019_attr17_train_attackAttr_embds_path,
    #     id_embeds_path=ASVspoof2019_attr17_eval_attackAttr_embds_path,
    #     ood_embeds_path=ASVspoof2019_attr2_full_bonafide_attackAttr_embds_path,
    #     k=10)
    
    # OOD detection using KNN-indegree
    knn_indegree_scores(
        train_embeds_path=ASVspoof2019_attr17_train_attackCls_embds_path,
        id_embeds_path=ASVspoof2019_attr17_eval_attackCls_embds_path,
        ood_embeds_path=ASVspoof2019_attr2_full_bonafide_attackCls_embds_path, 
        k=2000,
        M=2000)
    
    
    """
    # OOD detection using KNN-weighted_indegree
    # knn_weighted_indegree_scores(
    #     train_embeds_path=ASVspoof2019_attr17_train_attackCls_embds_path,
    #     id_embeds_path=ASVspoof2019_attr17_eval_attackCls_embds_path,
    #     ood_embeds_path=ASVspoof2019_attr2_full_bonafide_attackCls_embds_path, 
    #     k=2000, 
    #     M=200)
    """