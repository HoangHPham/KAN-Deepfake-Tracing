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
    
    train_emdeddings_path = './data/ASVspoof2019_attr17_ST_embeddings/ST_RB_SSLAASIST/asvspoof2019_attr17_ST_RB_sslaasist_train_emb.npy'
    dev_emdeddings_path = './data/ASVspoof2019_attr17_ST_embeddings/ST_RB_SSLAASIST/asvspoof2019_attr17_ST_RB_sslaasist_dev_emb.npy'
    
    train_embeddings = load_CMembeddings(train_emdeddings_path)
    train_ground_truth = create_ground_truth(train_protocols_path)
    
    dev_embeddings = load_CMembeddings(dev_emdeddings_path)
    dev_ground_truth = create_ground_truth(dev_protocols_path)
    
    print("Shape of train embeddings:", train_embeddings.shape)
    print("Shape of dev embeddings:", dev_embeddings.shape)
    
    train_dataset = MultitaskDataset(train_embeddings, train_ground_truth)
    dev_dataset = MultitaskDataset(dev_embeddings, dev_ground_truth)
    
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False, pin_memory=True, num_workers=8)
    dev_loader = DataLoader(dev_dataset, batch_size=10, shuffle=False, drop_last=False, pin_memory=True, num_workers=8)
    
    print("Number of batches in train loader:", len(train_loader))
    print("Number of batches in dev loader:", len(dev_loader))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    task_name = "T7_AS7"
    
    in_dim = 160
    out_dim = 11
    hdim = [64, 32]
    model = emb_fully_1(idim=in_dim, hdim=hdim, odim=out_dim).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))
    
    log_dir = "probabilistic_detectors_ST_RB_SSLAASIST/" + task_name + '_[64_32]' 
    os.makedirs(log_dir, exist_ok=True)
    
    lr = 0.0001
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    n_epochs = 100
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'dev_loss': [],
        'dev_acc': []
    }
    
    best_acc = -1e9
    
    for epoch in range(n_epochs):
        print("\n====> training epoch {:03d} / {:03d}".format(epoch+1, n_epochs))
        
        train_avg_loss, train_avg_acc = train_phase(train_loader, model, loss_fn, optimizer, device, task_name)
        history['train_loss'].append(train_avg_loss)
        history['train_acc'].append(train_avg_acc)

        print("Training loss: {} - Training accuracy: {}".format(train_avg_loss, train_avg_acc))
        
        dev_avg_loss, dev_avg_acc = dev_phase(dev_loader, model, loss_fn, device, task_name)
        history['dev_loss'].append(dev_avg_loss)
        history['dev_acc'].append(dev_avg_acc)
        
        print("Dev loss: {} - Dev accuracy: {}".format(dev_avg_loss, dev_avg_acc))
        
        torch.save(model.state_dict(), os.path.join(log_dir, 'last.pt'))
        
        if dev_avg_acc > best_acc:
            best_acc = dev_avg_acc
            torch.save(model.state_dict(), os.path.join(log_dir, 'best.pt'))
            
    plot(history, log_dir)
    
    # save history
    with open(os.path.join(log_dir, "history.pkl"), "wb") as f:
        pickle.dump(history, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Training completed.")
    

def plot(history, log_dir):
    # Loss vs Epoch array
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['dev_loss'], label='Dev Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Training and Development Loss vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'loss_vs_epoch.png'))
    plt.close()
    
    # Accuracy vs Epoch array
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['dev_acc'], label='Dev Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Development Accuracy vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'accuracy_vs_epoch.png'))
    plt.close()
    
    
def train_phase(train_loader, model, loss_fn, optimizer, device, task_name="T1_AS1"):
    
    model.train()
    
    total_loss = 0.
    total_acc = 0.
    
    total_samples = 0
    
    for embeddings, ground_truth in tqdm(train_loader):
        
        embeddings = embeddings.to(device)
        task_labels = ground_truth[task_name].to(device)
        
        total_samples += embeddings.shape[0]
        
        output = model(embeddings)
        
        loss = loss_fn(output, task_labels)
        
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        preds = torch.argmax(F.softmax(output, dim=1), dim=-1)
        total_acc += (preds == task_labels).float().sum().item()
        
    avg_loss = total_loss / len(train_loader)
    avg_acc = total_acc / total_samples
    
    return avg_loss, avg_acc


def dev_phase(loader, model, loss_fn, device, task_name="T1_AS1"):

    model.eval()
    
    total_loss = 0.
    total_acc = 0.
    
    total_samples = 0
    
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
            
    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / total_samples
    
    return avg_loss, avg_acc 


if __name__ == "__main__":
    main()