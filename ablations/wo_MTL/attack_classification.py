
import os
import pickle
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
    
    train_protocols_path = './data/ASVspoof2019_attr17_cm_protocols/Train_ASVspoof19_attr17.txt'
    dev_protocols_path = './data/ASVspoof2019_attr17_cm_protocols/Dev_ASVspoof19_attr17.txt'
    eval_protocols_path = './data/ASVspoof2019_attr17_cm_protocols/Eval_ASVspoof19_attr17.txt'
    
    train_post_probs_path = './data/posterior_probabilities_ST_RB_AASIST/train_task_probs.npy'
    dev_post_probs_path = './data/posterior_probabilities_ST_RB_AASIST/dev_task_probs.npy'
    eval_post_probs_path = './data/posterior_probabilities_ST_RB_AASIST/eval_task_probs.npy'
    
    train_attack_labels = load_attack_labels(train_protocols_path)
    dev_attack_labels = load_attack_labels(dev_protocols_path)
    eval_attack_labels = load_attack_labels(eval_protocols_path)
    
    train_post_probs = np.load(train_post_probs_path)
    dev_post_probs = np.load(dev_post_probs_path)
    eval_post_probs = np.load(eval_post_probs_path)
    
    train_dataset = AttClsDataset(train_post_probs, train_attack_labels)
    dev_dataset = AttClsDataset(dev_post_probs, dev_attack_labels)
    eval_dataset = AttClsDataset(eval_post_probs, eval_attack_labels)
    
    gen = torch.Generator()
    gen.manual_seed(seed)
    
    trn_loader = DataLoader(train_dataset,
                            batch_size=16,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=gen,
                            collate_fn=None,
                            num_workers=8)
    
    dev_loader = DataLoader(dev_dataset,
                            batch_size=16,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True,
                            collate_fn=None,
                            num_workers=8)
    
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
    os.makedirs(save_exp_path, exist_ok=True)
    
    model = KAN(width=[50, 64, 32, 17], grid=7, k=5, auto_save=False, device=device, seed=seed)
    # model.module_inFeatures(ASVSpoof2019_Attr17_attack_attribute_structure)
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    
    num_epochs = 100
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'dev_loss': [],
        'dev_acc': []
    }
    
    best_dev_acc = -1e9
    
    for epoch in range(num_epochs):
        print(f"\n=== Epoch: {epoch+1}/{num_epochs} ===")
        train_loss, train_acc = train_phase(trn_loader, model, criterion, optimizer, device)
        print(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
        
        dev_loss, dev_acc = devNeval_phase(dev_loader, model, criterion, device)
        print(f"Dev loss: {dev_loss:.4f}, Dev acc: {dev_acc:.4f}")
        
        torch.save(model.state_dict(), f"{save_exp_path}/last.pt")
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), f"{save_exp_path}/best.pt")
            
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['dev_loss'].append(dev_loss)
        history['dev_acc'].append(dev_acc)
        
    plot(history, save_exp_path)
    
    # save history
    with open(f"{save_exp_path}/history.pkl", "wb") as f:
        pickle.dump(history, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Training completed.")
    

def plot(history, save_exp_path):
    
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
    axes = axes.flatten()
    
    axes[0].plot(epochs, history['train_loss'], label='Train Loss')
    axes[0].plot(epochs, history['dev_loss'], label='Dev Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_ylim(0, 1)
    
    axes[1].plot(epochs, history['train_acc'], label='Train Acc')
    axes[1].plot(epochs, history['dev_acc'], label='Dev Acc')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_ylim(0, 1)
    
    fig.tight_layout()
    
    plt.savefig(f"{save_exp_path}/loss_and_accuracy.png")
    plt.show()

    
def train_phase(train_loader, model, criterion, optimizer, device):
    
    model.train()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for embeddings, attack_labels in tqdm(train_loader, desc="Training"):
        
        embeddings = embeddings.to(device, non_blocking=True)
        attack_labels = attack_labels.to(device, non_blocking=True)
        
        outputs = model(embeddings)
        
        loss = criterion(outputs, attack_labels)
        running_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(attack_labels.cpu().numpy())
        
    avg_loss = running_loss / len(train_loader)
    avg_acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, avg_acc


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
    
    return avg_loss, avg_acc
    
    
    
    
    
    
    
    

if __name__ == "__main__":
    main()
















