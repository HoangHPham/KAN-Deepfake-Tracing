import os
import sys
import random
import numpy as np

from sklearn.metrics import balanced_accuracy_score

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
    
    eval_protocols_path = './data/ASVspoof2019_attr17_cm_protocols/Eval_ASVspoof19_attr17.txt'
    
    eval_emdeddings_path = './data/ASVspoof2019_attr17_ST_embeddings/ST_RB_SSLAASIST/asvspoof2019_attr17_ST_RB_sslaasist_eval_emb.npy'
    
    eval_embeddings = load_CMembeddings(eval_emdeddings_path)
    eval_ground_truth = create_ground_truth(eval_protocols_path)
    
    eval_dataset = MultitaskDataset(eval_embeddings, eval_ground_truth)
    
    eval_loader = DataLoader(eval_dataset, batch_size=10, shuffle=False, drop_last=False, pin_memory=True, num_workers=8)
    
    print("Number of batches in eval loader:", len(eval_loader))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    task_name = "T7_AS7"
    
    log_dir = "probabilistic_detectors_ST_RB_SSLAASIST/" + task_name + '_[64_32]' 
    
    pretrained_model_path = log_dir + '/best.pt'
    
    in_dim = 160
    out_dim = 11
    hdim = [64, 32]
    model = emb_fully_1(idim=in_dim, hdim=hdim, odim=out_dim).to(device)
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    n_epochs = 100
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'dev_loss': [],
        'dev_acc': []
    }
    
    
    print("\n====> Starting evaluation")
    
    eval_avg_loss, eval_avg_acc, eval_balanced_acc = eval_phase(eval_loader, model, loss_fn, device, task_name)
    
    print("Eval loss: {} - Eval accuracy: {} - Eval balanced accuracy: {}".format(eval_avg_loss, eval_avg_acc, eval_balanced_acc))
            
    print("Evaluation completed.")


def eval_phase(loader, model, loss_fn, device, task_name="T1_AS1"):

    model.eval()
    
    total_loss = 0.
    total_acc = 0.
    
    total_samples = 0
    
    pred_labels = []
    true_labels = []
    
    with torch.no_grad():
        for embeddings, ground_truth in tqdm(loader):
            
            embeddings = embeddings.to(device)
            task_labels = ground_truth[task_name].to(device)
            
            total_samples += embeddings.shape[0]
            
            output = model(embeddings)
            
            loss = loss_fn(output, task_labels)
            total_loss += loss.item()
            
            preds = torch.argmax(F.softmax(output, dim=1), dim=-1)
            total_acc += (preds == task_labels).float().sum().item()
            
            pred_labels.extend(preds.cpu().numpy())
            true_labels.extend(task_labels.cpu().numpy())
            
    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / total_samples
    
    balanced_acc = balanced_accuracy_score(true_labels, pred_labels)
    
    return avg_loss, avg_acc, balanced_acc 


if __name__ == "__main__":
    main()