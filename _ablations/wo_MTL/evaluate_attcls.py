
import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score

import torch
from torch.utils.data import DataLoader

from create_ground_truth import load_attack_labels
from data_utils import AttClsDataset, ASVSpoof2019_Attr17_attack_attribute_structure
from utils import seed_worker, set_seed

from kan_simplify import *


def main():
    
    seed = 42
    set_seed(seed)
    
    eval_protocols_path = './data/ASVspoof2019_attr17_cm_protocols/Eval_ASVspoof19_attr17.txt'
    
    eval_post_probs_path = './data/posterior_probabilities_ST_RB_AASIST/eval_task_probs.npy'
    
    eval_attack_labels = load_attack_labels(eval_protocols_path)
    
    eval_post_probs = np.load(eval_post_probs_path)
    
    eval_dataset = AttClsDataset(eval_post_probs, eval_attack_labels)
    
    
    eval_loader = DataLoader(eval_dataset,
                             batch_size=16,
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True,
                             collate_fn=None,
                             num_workers=8)
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    exp_name = 'KAN_[50_64_32_17]_G7_k5'
    save_exp_path = os.path.join("exp_results", exp_name)
    
    model = KAN(width=[50, 64, 32, 17], grid=7, k=5, auto_save=False, device=device, seed=seed)
    # model.module_inFeatures(ASVSpoof2019_Attr17_attack_attribute_structure)
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    pretrained_weights_path = f"{save_exp_path}/best.pt"
    model.load_state_dict(torch.load(pretrained_weights_path, map_location=device))
    model.to(device)
    
    print("Starting evaluation on the evaluation set...")
    eval_loss, eval_acc, eval_balanced_acc = devNeval_phase(eval_loader, model, criterion, device)
    print(f"Eval loss: {eval_loss}, Eval acc: {eval_acc}, Eval balanced acc: {eval_balanced_acc}")
    
    print("Evaluation completed.")
    

def devNeval_phase(loader, model, criterion, device):
    
    model.eval()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for embeddings, attack_labels in tqdm(loader, desc="Evaluating"):
            
            embeddings = embeddings.to(device, non_blocking=True)
            attack_labels = attack_labels.to(device, non_blocking=True)
            outputs = model(embeddings)
            
            loss = criterion(outputs, attack_labels)
            running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(attack_labels.cpu().numpy())
            
    avg_loss = running_loss / len(loader)
    avg_acc = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    
    return avg_loss, avg_acc, balanced_acc
    
    

if __name__ == "__main__":
    main()
