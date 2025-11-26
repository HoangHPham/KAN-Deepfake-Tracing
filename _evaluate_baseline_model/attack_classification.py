
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm

import shap

from create_ground_truth import load_attack_labels


def compute_shap_and_fi(clf: DecisionTreeClassifier,
                        X_train: np.ndarray,
                        X_eval: np.ndarray):
    
    explainer = shap.TreeExplainer(
        clf,
        data=X_train,                   # used for background/expected value
        model_output="raw",     # explanations sum to predicted probability per class
        feature_perturbation="interventional"
    )

    shap_FI = explainer.shap_values(X_eval, check_additivity=True)
    
    # S: (N, F, C)
    gfi = np.abs(shap_FI).mean(axis=(0, 2))          # (F,)
    gfi_norm = gfi / (gfi.sum() + 1e-12)
    
    gfi_by_class = np.abs(shap_FI).mean(axis=0).T    # (F, C) -> transpose -> (C, F) = (17, 50)
    gfi_by_class_norm = gfi_by_class / (gfi_by_class.sum(axis=1, keepdims=True) + 1e-12)
    
    return gfi_norm, gfi_by_class_norm


def compute_shap_and_fi_logreg(clf: LogisticRegression,
                        X_train: np.ndarray,
                        X_eval: np.ndarray,
                        k_bg: int = 64):
    
    bg = shap.kmeans(X_train, k=k_bg) 
    masker = shap.maskers.Independent(bg.data)
    
    explainer = shap.Explainer(
        clf,
        masker=masker,
        algorithm="linear",
        link=shap.links.logit,   # explain in log-odds; sums add in this space
    )

    shap_FI = explainer.shap_values(X_eval)
    
    # S: (N, F, C)
    gfi = np.abs(shap_FI).mean(axis=(0, 2))          # (F,)
    gfi_norm = gfi / (gfi.sum() + 1e-12)
    
    gfi_by_class = np.abs(shap_FI).mean(axis=0).T    # (F, C) -> transpose -> (C, F) = (17, 50)
    gfi_by_class_norm = gfi_by_class / (gfi_by_class.sum(axis=1, keepdims=True) + 1e-12)
    
    return gfi_norm, gfi_by_class_norm


def main():
    
    in_vars=["Text (inputs)", "Speech_human (inputs)", "Speech_TTS (inputs)",
         "NLP (processor)", "WORLD (processor)", "LPCC/MFCC (processor)", "CNN+bi-RNN (processor)", "ASR (processor)", "MFCC/i-vector (processor)",
         "HMM (duration)", "FF (duration)", "RNN (duration)", "Attention (duration)", "DTW (duration)", "None (duration)", 
         "AR-RNN (conversion)", "FF (conversion)", "CART (conversion)", "VAE (conversion)", "GMM-UBM (conversion)", "RNN (conversion)", "AR-RNN+CNN (conversion)", "Moment-match (conversion)", "linear (conversion)",
         "VAE (speaker)", "one-hot (speaker)", "d-vector_RNN (speaker)", "PLDA (speaker)", "None (speaker)", 
         "MCC-F0 (outputs)", "MCC-F0-BAP (outputs)", "MFCC-F0 (outputs)", "MCC-F0-AP (outputs)", "LPC (outputs)", "MCC-F0-BA (outputs)", "mel-spec (outputs)", "F0+ling (outputs)", "MCC (outputs)", "MFCC (outputs)",
         "WaveNet (waveform)", "WORLD (waveform)", "Concat (waveform)", "SpecFiltOLA (waveform)", "NeuralFilt (waveform)", "Vocaine (waveform)", "WaveRNN (waveform)", "GriffinLim (waveform)", "WaveFilt (waveform)", "STRAIGHT (waveform)", "MFCCvoc (waveform)"
        ]
    
    train_protocols_path = './data/ASVspoof2019_attr17_cm_protocols/Train_ASVspoof19_attr17.txt'
    dev_protocols_path = './data/ASVspoof2019_attr17_cm_protocols/Dev_ASVspoof19_attr17.txt'
    eval_protocols_path = './data/ASVspoof2019_attr17_cm_protocols/Eval_ASVspoof19_attr17.txt'
    
    train_post_probs_path = './posterior_probabilities_ST_RB_SSLAASIST/train_task_probs.npy'
    dev_post_probs_path = './posterior_probabilities_ST_RB_SSLAASIST/dev_task_probs.npy'
    eval_post_probs_path = './posterior_probabilities_ST_RB_SSLAASIST/eval_task_probs.npy'
    
    train_attack_labels = load_attack_labels(train_protocols_path)
    dev_attack_labels = load_attack_labels(dev_protocols_path)
    eval_attack_labels = load_attack_labels(eval_protocols_path)
    
    train_post_probs = np.load(train_post_probs_path)
    dev_post_probs = np.load(dev_post_probs_path)
    eval_post_probs = np.load(eval_post_probs_path)
    
    state = 3000
    # clf = DecisionTreeClassifier(max_depth=20, random_state=state)
    # clf = GaussianNB()
    # clf = LogisticRegression(max_iter=1000, random_state=state)
    clf = svm.SVC(kernel='rbf', max_iter=1000, probability=True, random_state=state)
    
    clf = clf.fit(train_post_probs, train_attack_labels)
    dev_pred = clf.predict(dev_post_probs)
    eval_pred = clf.predict(eval_post_probs)
    
    print("\n=== Dev set results ===")
    print("Accuracy: ", accuracy_score(dev_attack_labels, dev_pred))
    # print(classification_report(dev_attack_labels, dev_pred, digits=4))
    # print("Confusion Matrix:\n", confusion_matrix(dev_attack_labels, dev_pred))
    
    print("\n=== Eval set results ===")
    print("Accuracy: ", accuracy_score(eval_attack_labels, eval_pred))
    print("Balanced accuracy: ", balanced_accuracy_score(eval_attack_labels, eval_pred))
    
    
    
    # print(classification_report(eval_attack_labels, eval_pred, digits=4))
    # print("Confusion Matrix:\n", confusion_matrix(eval_attack_labels, eval_pred))
    
    """ SHAP feature importance
    gfi_norm, gfi_by_class_norm = compute_shap_and_fi_logreg(
        clf=clf,
        X_train=train_post_probs,
        X_eval=eval_post_probs
    )

    print("\nShapes:")
    print("  Global FI (overall):", gfi_norm.shape)   # (50,)
    print("  Global FI (per class):", gfi_by_class_norm.shape)  # (17, 50)
    
    print(gfi_norm)
    
    df = pd.DataFrame({'Feature': np.array(in_vars), 'Score': gfi_norm})
    df_sorted = df.sort_values(by='Score', ascending=False)

    # Plot
    plt.figure(figsize=(30, 6))
    plt.bar(df_sorted['Feature'], df_sorted['Score'])
    plt.xticks(rotation=90)
    plt.ylabel('Feature Score')
    plt.title('Feature Scores')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('feature_importance_overall.png')
    plt.show()

    plot_inFeature_importance(gfi_by_class_norm, 0, in_vars)
    """

def plot_inFeature_importance(shap_scores, outNode_id, in_vars):
    scores = shap_scores[outNode_id]
    
    print(scores)

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
    plt.savefig('feature_importance_A0{}.png'.format(outNode_id+1))
    plt.show()

if __name__ == "__main__":
    main()
















