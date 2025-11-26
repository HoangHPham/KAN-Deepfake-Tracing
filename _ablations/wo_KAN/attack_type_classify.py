
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

from create_ground_truth import load_attack_labels


def main():
    
    train_protocols_path = './data/ASVspoof2019_attr17_cm_protocols/Train_ASVspoof19_attr17.txt'
    dev_protocols_path = './data/ASVspoof2019_attr17_cm_protocols/Dev_ASVspoof19_attr17.txt'
    eval_protocols_path = './data/ASVspoof2019_attr17_cm_protocols/Eval_ASVspoof19_attr17.txt'
    
    train_attattr_embds_path = './extracted_embds/ST_RB_AASIST_MTL_pf/ASVspoof2019_attr17_train_AblateKAN_attackAttr_embeddings.npy'
    dev_attattr_embds_path = './extracted_embds/ST_RB_AASIST_MTL_pf/ASVspoof2019_attr17_dev_AblateKAN_attackAttr_embeddings.npy'
    eval_attattr_embds_path = './extracted_embds/ST_RB_AASIST_MTL_pf/ASVspoof2019_attr17_eval_AblateKAN_attackAttr_embeddings.npy'
    
    train_attack_labels = load_attack_labels(train_protocols_path)
    dev_attack_labels = load_attack_labels(dev_protocols_path)
    eval_attack_labels = load_attack_labels(eval_protocols_path)
    
    train_attrattr_embds = np.load(train_attattr_embds_path)
    dev_attrattr_embds = np.load(dev_attattr_embds_path)
    eval_attrattr_embds = np.load(eval_attattr_embds_path)
    
    state = 3000
    # clf = DecisionTreeClassifier(max_depth=20, random_state=state)
    # clf = GaussianNB()
    # clf = LogisticRegression(max_iter=1000, random_state=state)
    clf = svm.SVC(kernel='rbf', max_iter=1000, probability=True, random_state=state)
    
    clf = clf.fit(train_attrattr_embds, train_attack_labels)
    dev_pred = clf.predict(dev_attrattr_embds)
    eval_pred = clf.predict(eval_attrattr_embds)
    
    print("\n=== Dev set results ===")
    print("Accuracy: ", accuracy_score(dev_attack_labels, dev_pred))
    # print(classification_report(dev_attack_labels, dev_pred, digits=4))
    # print("Confusion Matrix:\n", confusion_matrix(dev_attack_labels, dev_pred))
    
    print("\n=== Eval set results ===")
    print("Accuracy: ", accuracy_score(eval_attack_labels, eval_pred))
    print("Balanced accuracy: ", balanced_accuracy_score(eval_attack_labels, eval_pred))
    
    
    # print(classification_report(eval_attack_labels, eval_pred, digits=4))
    # print("Confusion Matrix:\n", confusion_matrix(eval_attack_labels, eval_pred))
    
if __name__ == "__main__":
    main()