import os
import sys
import random
import numpy as np

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from create_ground_truth import create_ground_truth, load_CMembeddings
from data_utils import MultitaskDataset
from emb_model import emb_fully_1

import matplotlib.pyplot as plt
import pickle

from tqdm.auto import tqdm


def set_seed(seed: int = 42):
    random.seed(seed)                    
    np.random.seed(seed)                 
    torch.manual_seed(seed)               
    torch.cuda.manual_seed(seed)          
    torch.cuda.manual_seed_all(seed)      

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    
    seed = 42
    set_seed(seed)
    
    train_protocols_path = './data/ASVspoof2019_attr17_cm_protocols/Train_ASVspoof19_attr17.txt'
    dev_protocols_path = './data/ASVspoof2019_attr17_cm_protocols/Dev_ASVspoof19_attr17.txt'
    eval_protocols_path = './data/ASVspoof2019_attr17_cm_protocols/Eval_ASVspoof19_attr17.txt'
    
    train_emdeddings_path = './data/ASVspoof2019_attr17_ST_embeddings/ST_RB_SSLAASIST/asvspoof2019_attr17_ST_RB_sslaasist_train_emb.npy'
    dev_emdeddings_path = './data/ASVspoof2019_attr17_ST_embeddings/ST_RB_SSLAASIST/asvspoof2019_attr17_ST_RB_sslaasist_dev_emb.npy'
    eval_emdeddings_path = './data/ASVspoof2019_attr17_ST_embeddings/ST_RB_SSLAASIST/asvspoof2019_attr17_ST_RB_sslaasist_eval_emb.npy'
    
    train_embeddings = load_CMembeddings(train_emdeddings_path)
    train_ground_truth = create_ground_truth(train_protocols_path)
    
    dev_embeddings = load_CMembeddings(dev_emdeddings_path)
    dev_ground_truth = create_ground_truth(dev_protocols_path)
    
    eval_embeddings = load_CMembeddings(eval_emdeddings_path)
    eval_ground_truth = create_ground_truth(eval_protocols_path)
    
    train_dataset = MultitaskDataset(train_embeddings, train_ground_truth)
    dev_dataset = MultitaskDataset(dev_embeddings, dev_ground_truth)
    eval_dataset = MultitaskDataset(eval_embeddings, eval_ground_truth)
    
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=False, drop_last=False, pin_memory=True, num_workers=8)
    dev_loader = DataLoader(dev_dataset, batch_size=10, shuffle=False, drop_last=False, pin_memory=True, num_workers=8)
    eval_loader = DataLoader(eval_dataset, batch_size=10, shuffle=False, drop_last=False, pin_memory=True, num_workers=8)
    
    print("Number of batches in train loader:", len(train_loader))
    print("Number of batches in dev loader:", len(dev_loader))
    print("Number of batches in eval loader:", len(eval_loader))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    
    tasks = ["T1_AS1", "T2_AS2", "T3_AS3", "T4_AS4", "T5_AS5", "T6_AS6", "T7_AS7"]
    
    out_dims = [3, 6, 6, 9, 5, 10, 11]
    
    save_folder = 'posterior_probabilities_ST_RB_SSLAASIST'
    os.makedirs(save_folder, exist_ok=True)
    
    print("\n\n====> Extracting probabilities for train set...")
    train_task_probs = extract_probs_full(train_loader, tasks, out_dims, device)
    print("Shape of train_task_probs:", train_task_probs.shape)
    np.save(os.path.join(save_folder, 'train_task_probs.npy'), train_task_probs)
    
    print("\n\n====> Extracting probabilities for dev set...")
    dev_task_probs = extract_probs_full(dev_loader, tasks, out_dims, device)
    print("Shape of dev_task_probs:", dev_task_probs.shape)
    np.save(os.path.join(save_folder, 'dev_task_probs.npy'), dev_task_probs)
    
    print("\n\n====> Extracting probabilities for eval set...")
    eval_task_probs = extract_probs_full(eval_loader, tasks, out_dims, device)
    print("Shape of eval_task_probs:", eval_task_probs.shape)
    np.save(os.path.join(save_folder, 'eval_task_probs.npy'), eval_task_probs)
    

def extract_probs_full(loader, tasks, out_dims, device):
    
    all_task_probs = []
    for task_name, out_dim in zip(tasks, out_dims):
        print("=> Task: ", task_name)
        
        log_dir = "probabilistic_detectors_ST_RB_SSLAASIST/" + task_name + '_[64_32]' 
        pretrained_model_path = log_dir + '/best.pt'
        
        in_dim = 160
        hdim = [64, 32]
        model = emb_fully_1(idim=in_dim, hdim=hdim, odim=out_dim).to(device)
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        
        task_probs = extract_probs_per_task(loader, model, device)
        
        all_task_probs.append(task_probs)
    
    all_task_probs = np.concatenate(all_task_probs, axis=1)
    return all_task_probs
    

def extract_probs_per_task(loader, model, device):

    model.eval()
    
    all_probs = []
    with torch.no_grad():
        for embeddings, _ in tqdm(loader):
            
            embeddings = embeddings.to(device)
            
            output = model(embeddings)
            
            probs = F.softmax(output, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            
    all_probs = np.concatenate(all_probs, axis=0)
    return all_probs


if __name__ == "__main__":
    main()