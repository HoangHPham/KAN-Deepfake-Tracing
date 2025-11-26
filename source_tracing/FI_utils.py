import os
import math
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
from itertools import combinations

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import balanced_accuracy_score

from tqdm.auto import tqdm


class AttClsDataset(Dataset):
    def __init__(self, embeddings, attack_labels):
        """
        Initialize the dataset with embeddings and ground truth labels.
        Args:
            embeddings (np.ndarray): Array of embeddings.
            attack_labels  (np.ndarray): Array of attack labels.
        """
        self.embeddings = embeddings
        self.attack_labels = attack_labels
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, index):
        """
        Get the item at the specified index.
        Args:
            idx (int): Index of the item to retrieve.
        Returns:
            tuple: (embedding, attack_label) for the specified index.
        """
        embedding = torch.tensor(self.embeddings[index], dtype=torch.float32)
        attack_label = torch.tensor(self.attack_labels[index], dtype=torch.long)
        
        return embedding, attack_label
    


def load_CMembeddings(CMembeddings_path):
    """
    Load 160-dimensional AASIST's embeddings from the specified path.
    Args:
        embeddings_path (str): Path to the embeddings file.
    Returns:
        np.ndarray: Loaded embeddings as a numpy array.
    """
    if not os.path.exists(CMembeddings_path):
        raise FileNotFoundError(f"Embeddings file not found at {CMembeddings_path}")
    embeddings = np.load(CMembeddings_path)
    if embeddings.ndim != 2 or embeddings.shape[1] != 160:
        raise ValueError(f"Embeddings should be a 2D array with 160 dimensions, got shape {embeddings.shape}")
    
    return embeddings.astype(np.float32)  # Ensure the embeddings are in float32 format


def load_protocol_data(protocols_path):
    """
    Load protocol data from the specified path.
    Args:
        protocols_path (str): Path to the protocols file.
    Returns:
        pd.DataFrame: Loaded protocol data as a pandas DataFrame.
    """
    if not os.path.exists(protocols_path):
        raise FileNotFoundError(f"Protocols file not found at {protocols_path}")
    df = pd.read_csv(protocols_path, sep=" ", header=None)
    
    return df


def get_labels(source):

    # data based on "Table 1: Summary of LA spoofing systems." in "ASVspoof 2019: a large-scale public database of synthetized, converted and replayed speech"
    
    # the data is not balanced when it comes to the attributes (e.g. Text(input) has more samples than MCC-F0(output))
    data = {
            "A01":      [[0], [0], [0], [0], [0], [0], [0], [0]],
            "A02":      [[1], [0], [0], [0], [0], [0], [1], [1]],
            "A03":      [[2], [0], [0], [1], [1], [1], [1], [1]],
            "A04-16":   [[3], [0], [0], [5], [2], [4], [2], [2]],
            "A05":      [[4], [1], [1], [5], [3], [1], [3], [1]],
            "A06-19":   [[5], [1], [2], [5], [4], [4], [4], [3]],

            "A07":      [[6], [0], [0], [2], [5], [1], [5], [1]],
            "A08":      [[7], [0], [0], [0], [0], [1], [0], [4]],
            "A09":      [[8], [0], [0], [2], [5], [1], [0], [5]],
            "A10":      [[9], [0], [3], [3], [6], [2], [6], [6]], 
            "A11":      [[10], [0], [3], [3], [6], [2], [6], [7]],
            "A12":      [[11], [0], [0], [2], [5], [1], [7], [0]],
            "A13":      [[12], [2], [1], [4], [7], [4], [8], [8]],
            "A14":      [[13], [2], [4], [5], [5], [4], [1], [9]],
            "A15":      [[14], [2], [4], [5], [5], [4], [0], [0]],
            "A17":      [[15], [1], [1], [5], [3], [1], [0], [8]],
            "A18":      [[16], [1], [5], [5], [8], [3], [9], [10]],
        } 
    
    attribute_sets = ["AS1", "AS2", "AS3", "AS4", "AS5", "AS6", "AS7"]
    
    labels = {}
    for attack_type, ys in data.items():
        labels[attack_type] = {
            "Attack_id": ys[0],
            **{f'T{i+1}_{attribute_sets[i]}': ys[i+1] for i in range(len(attribute_sets))},
        }
    
    return labels[source]


def create_ground_truth(protocols_path):
    """
    Create ground truth for multi-task learning based on embeddings and protocols.
    Args:
        embeddings_path (str): Path to the embeddings file.
        protocols_path (str): Path to the protocols file.
    """
    protocols_data = load_protocol_data(protocols_path)
    
    ground_truth = []
    for index, row in protocols_data.iterrows():
        source = row[3]
        attributes = get_labels(source)
        ground_truth.append(attributes)
    ground_truth = np.array(ground_truth, dtype=object)
    return ground_truth


def load_attack_labels(protocols_path):
    protocols_data = load_protocol_data(protocols_path)
    
    ground_truth = []
    for index, row in protocols_data.iterrows():
        source = row[3]
        attributes = get_labels(source)
        ground_truth.append(attributes['Attack_id'][0])
    ground_truth = np.array(ground_truth, dtype=np.int64)
    return ground_truth


def rank_feature_scores(feature_scores):
    uniq_desc = np.unique(feature_scores)[::-1]
    rank_map = {v: i + 1 for i, v in enumerate(uniq_desc)}
    return np.array([rank_map[v] for v in feature_scores], dtype=int)


def spearman_rank_corr(r1, r2, n=None):
    if n == None:
        n = len(r1)
    d = r1 - r2
    return float(1 - 6 * np.sum(d*d) / (n * (n*n - 1)))


def spearman_pairwise(rank_vectors, task='entire'):

    if task == 'per_attack':
        n = 7
    elif task == 'entire':
        n = 50
        
    m = len(rank_vectors)

    M = np.eye(m, dtype=float)
    pairwise = {}

    for (i, r1), (j, r2) in combinations(enumerate(rank_vectors), 2):
        rho = spearman_rank_corr(r1, r2, n=n)
        M[i, j] = M[j, i] = rho

    return M


def plot_corr_matrix(M, names=None, title="Spearman rank correlation"):
    m = len(M)
    if names is None: names = [f"V{i+1}" for i in range(m)]
    df = pd.DataFrame(M, index=names, columns=names)
    # mask = np.tril(np.ones_like(df, dtype=bool), k=-1)  # hide strictly lower
    ax = sns.heatmap(df, vmin=-1, vmax=1, annot=True, fmt=".2f",
                     cbar_kws={'label': 'ρ'}, cmap='BuGn')
    ax.set_title(title)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.tight_layout()
    plt.show()
    
    
def plot_corr_grid(M_list, titles=None, names=None, upper_only=True,
                   cmap="YlGn", annot=True, fmt=".2f"):
    
    k = len(M_list)
    N = M_list[0].shape[0]
    assert all(M.shape == (N, N) for M in M_list), "All matrices must be same size."

    if titles is None:
        titles = [f"A{i+1:02d}" for i in range(k)]
    if names is None:
        names = [str(i+1) for i in range(N)]

    # choose a compact grid
    cols = min(6, k)                    # up to 5 across; adjust if you prefer
    rows = math.ceil(k / cols)

    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols*3.2, rows*3.2),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    vmin, vmax = -1.0, 1.0
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for ax, M, title in zip(axes, M_list, titles):
        df = pd.DataFrame(M, index=names, columns=names)
        mask = np.tril(np.ones_like(df, dtype=bool), k=-1) if upper_only else None
        sns.heatmap(df, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                    square=True, mask=mask, annot=annot, fmt=fmt,
                    cbar=False)
        ax.set_title(title, pad=6, fontsize=10)
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, labelsize=8)

    # hide any unused axes
    for ax in axes[len(M_list):]:
        ax.axis("off")

    # one shared colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:len(M_list)], fraction=0.03, pad=0.01)
    cbar.set_label("ρ")

    plt.show()
    

def get_FI_per_attack(kan_model):

    all_scores = []
    for i in range(17):
        scores = kan_model.attribute(1, i, plot=False).detach().cpu().numpy()
        all_scores.append(scores)

    return np.array(all_scores)


def get_fi_per_bs(loader, model):
    
    model.eval()

    per_samples_features_scores = []
    with torch.no_grad():
        for embeddings, _ in tqdm(loader):
            
            embeddings = embeddings.to(device)
            
            AttackCls_outputs = model(embeddings)

            feature_scores = model.feature_score

            feature_scores = feature_scores.detach().cpu().numpy()
        
            per_samples_features_scores.append(feature_scores)

    per_samples_features_scores = np.array(per_samples_features_scores)

    std_FI_by_samples = np.std(per_samples_features_scores, axis=0)

    return std_FI_by_samples 


def get_fi_per_bs_by_attack(loader, kan_model, device):
    
    kan_model.eval()

    per_samples_features_scores = []
    with torch.no_grad():
        for embeddings, _ in tqdm(loader):
            
            embeddings = embeddings.to(device)
            
            AttackCls_outputs = kan_model(embeddings)

            feature_scores = get_FI_per_attack(kan_model)

            per_samples_features_scores.append(feature_scores)

    per_samples_features_scores = np.array(per_samples_features_scores)

    if per_samples_features_scores.shape[0] == 1:
        return per_samples_features_scores[0]

    std_FI_by_samples = np.std(per_samples_features_scores, axis=0)

    return std_FI_by_samples 


def plot_inFeature_importance(scores, outNode_id):
    scores = scores[outNode_id]

    # Create a DataFrame for sorting
    df = pd.DataFrame({'Feature': np.array(in_vars), 'Score': scores})
    df_sorted = df.sort_values(by='Score', ascending=False)
    
    # Plot
    plt.figure(figsize=(30, 6))
    plt.bar(df_sorted['Feature'], df_sorted['Score'])
    plt.xticks(rotation=90)
    plt.ylabel('Feature Score')
    plt.title('Input feature importance for A0{}'.format(outNode_id+1))
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    
    
def plot_cfm(cm, class_names):
    # Normalize per row (for per-class accuracy view)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Plot
    plt.figure(figsize=(12, 12))
    sns.heatmap(cm_normalized, annot=True, cmap="BuPu",
                xticklabels=class_names,
                yticklabels=class_names,
                fmt=".2f")
    
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    # plt.title("Normalized Confusion Matrix (Per class Accuracy - %)")
    plt.show()
    
    
def get_rank_perAtk_multi_runs(attack=0, list_runs=None):
    
    runs_scores = np.array([rank_feature_scores(x[attack]) for x in list_runs]) 

    M = spearman_pairwise(runs_scores, task='per_attack')

    return M


def eval_AttackCls(loader, model, device):

    model.eval()

    total_acc = 0.0
    
    all_true = []
    all_pred = []
    
    total = 0

    with torch.no_grad():
        for embeddings, attack_labels in tqdm(loader):
            
            embeddings = embeddings.to(device)
            attack_labels = attack_labels.to(device)
            
            AttackCls_outputs = model(embeddings)
            
            total += embeddings.shape[0]
            
            AttackCls_probs = F.softmax(AttackCls_outputs, dim=-1) # [B, n_attack_types]
            pred_AttackCls = AttackCls_probs.argmax(dim=-1)  # [B]
            AttackCls_acc = (pred_AttackCls == attack_labels).float().sum().item()  # acc for 1 batch of #n_spoof_samples_in_a_batch
            total_acc += AttackCls_acc
            
            # used for balanced accuracy
            pred_AttackCls = pred_AttackCls.cpu().numpy()
            true_AttackCls = attack_labels.cpu().numpy()
            all_pred.extend(pred_AttackCls)
            all_true.extend(true_AttackCls)
            
    avg_acc = total_acc / total
    
    # balanced accuracy
    avg_balanced_acc = balanced_accuracy_score(all_true, all_pred)

    attCls_cfsMatrix = confusion_matrix(all_true, all_pred)
    attCls_clsReport = classification_report(all_true, all_pred, digits=3)
    
    return avg_acc, avg_balanced_acc, attCls_cfsMatrix, attCls_clsReport 