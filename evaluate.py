import sys
import random
import numpy as np

from sklearn.metrics import balanced_accuracy_score

import yaml

import warnings

from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from data_utils import fetch_protocol, Dataset_ASVspoof2019_attr17, ASVSpoof2019_Attr17_attack_attribute_structure
from utils import set_seed
from model import CMMTLKAN

warnings.filterwarnings("ignore", category=FutureWarning)


def eval_phase(loader, model, device):

    model.eval()
    
    total_loss = {
        **{f"T{i+1}_AS{i+1}": 0.0 for i in range(7)},
        'AttackAttr_loss': 0.0,
        'AttackCls_loss': 0.0,
        'running_loss': 0.0
    }
    
    total_acc = {
        **{f"T{i+1}_AS{i+1}": 0.0 for i in range(7)},
        'AttackAttr_acc': 0.0,
        'AttackCls_acc': 0.0,
    }
    
    all_true = {
        **{f"T{i+1}_AS{i+1}": [] for i in range(7)},
        'AttackCls': []
    }
    all_pred = {
        **{f"T{i+1}_AS{i+1}": [] for i in range(7)},
        'AttackCls': []
    }
    
    total_spoof_samples = 0
    
    with torch.no_grad():
        for waveform, ground_truth in tqdm(loader):
            
            waveform = waveform.to(device)
            
            attack_labels = ground_truth["Attack_id"].to(device) # [B]
            task_labels = {
                task: labels.to(device)
                for task, labels in ground_truth.items()
                if task != 'Attack_id'
            } # dict: {task_name: [B]} 
            
            GWMTL_outputs, AttackCls_outputs = model(waveform)
            
            total_spoof_samples += waveform.shape[0]
            
            multi_task_losses = []
            multi_task_accs = []
            for task_name, task_logits in GWMTL_outputs.items():
                
                task_loss = F.cross_entropy(
                    task_logits, 
                    task_labels[task_name],
                    reduction='none',
                    label_smoothing=0.05
                ) # [B] # loss for per sample
                
                total_loss[task_name] += task_loss.sum().item()  # total loss of 1 batch on spoof samples
                multi_task_losses.append(task_loss)
                
                task_probs = F.softmax(task_logits, dim=-1)  # [B, n_classes]
                pred_task = task_probs.argmax(dim=-1) # [B]
                task_acc = (pred_task == task_labels[task_name]).float().sum().item() # acc for 1 batch of #n_spoof_samples_in_a_batch
                multi_task_accs.append(task_acc)
                total_acc[task_name] += task_acc
                
                # balanced accuracy
                pred_task = pred_task.cpu().numpy()
                true_task = task_labels[task_name].cpu().numpy()
                all_pred[task_name].extend(pred_task)
                all_true[task_name].extend(true_task)
                
                
            # sum losses over all tasks
            multi_task_losses = torch.stack(multi_task_losses, dim=1).sum(dim=1) # [#n_spoof_samples_in_a_batch]
            total_loss['AttackAttr_loss'] += multi_task_losses.sum().item() # total loss of 1 batch on spoof samples
            
            # sum accuracies over all tasks
            multi_task_accs = torch.tensor(multi_task_accs, dtype=torch.float32, device=device)
            total_acc['AttackAttr_acc'] += multi_task_accs.sum().item()
                    
            AttackCls_loss = F.cross_entropy(
                AttackCls_outputs,
                attack_labels,
                reduction='none'
            )  # [B] # loss for per sample  
                
            total_loss['AttackCls_loss'] += AttackCls_loss.sum().item()  # total loss of 1 batch on spoof samples
            
            AttackCls_probs = F.softmax(AttackCls_outputs, dim=-1) # [B, n_attack_types]
            pred_AttackCls = AttackCls_probs.argmax(dim=-1)  # [B]
            AttackCls_acc = (pred_AttackCls == attack_labels).float().sum().item()  # acc for 1 batch of #n_spoof_samples_in_a_batch
            total_acc['AttackCls_acc'] += AttackCls_acc
            
            # balanced accuracy
            pred_AttackCls = pred_AttackCls.cpu().numpy()
            true_AttackCls = attack_labels.cpu().numpy()
            all_pred['AttackCls'].extend(pred_AttackCls)
            all_true['AttackCls'].extend(true_AttackCls)
            
            running_loss = multi_task_losses.mean() + AttackCls_loss.mean() # total loss for 1 batch
            
            total_loss['running_loss'] += running_loss.item()  # accumulate total loss
            
    avg_loss = {
        **{task_name: loss_value / total_spoof_samples for task_name, loss_value in total_loss.items() if task_name != "running_loss" and task_name != "AttackCls_loss"}
    }
    avg_loss['AttackCls_loss'] = total_loss['AttackCls_loss'] / total_spoof_samples
    avg_loss["running_loss"] = total_loss["running_loss"] / len(loader)
    
    avg_acc = {
        **{task_name: acc / total_spoof_samples for task_name, acc in total_acc.items() if task_name != "AttackCls_acc" and task_name != "AttackAttr_acc"}
    }
    avg_acc["AttackAttr_acc"] = total_acc['AttackAttr_acc'] / (total_spoof_samples * 7)
    avg_acc["AttackCls_acc"] = total_acc['AttackCls_acc'] / total_spoof_samples
    avg_acc["total_acc"] = (avg_acc["AttackAttr_acc"] + avg_acc["AttackCls_acc"]) / 2
    
    # balanced accuracy
    balanced_acc = {}
    for task_name in all_true.keys():
        balanced_acc[task_name] = balanced_accuracy_score(all_true[task_name], all_pred[task_name])
    
    balanced_acc["AttackAttr_bal_acc"] = sum(balanced_acc[t] for t in balanced_acc.keys() if t != 'AttackCls') / 7
    balanced_acc["total_bal_acc"] = (balanced_acc["AttackAttr_bal_acc"] + balanced_acc["AttackCls"]) / 2
    
    return avg_loss, avg_acc, balanced_acc 


if __name__ == "__main__":
    
    data_config_yaml_file = './data/ASVspoof2019_attr17_cm.yaml'
    eval_config_yaml_file = './config/evaluate.yaml'
    
    with open(data_config_yaml_file) as f:
        data_config = yaml.load(f, Loader=yaml.SafeLoader)

    with open(eval_config_yaml_file) as f:
        eval_config = yaml.load(f, Loader=yaml.SafeLoader)
    
    set_seed(eval_config)

    eval_dataset_path = data_config['eval_dataset_path']
    eval_protocols_path = data_config['eval_protocol_path']
    
    eval_flac_files, eval_ground_truths = fetch_protocol(eval_protocols_path)
    
    eval_dataset = Dataset_ASVspoof2019_attr17(
                                                flac_files=eval_flac_files, 
                                                ground_truths=eval_ground_truths, 
                                                base_dir=eval_dataset_path,
                                                phase='eval',
                                                trim_silence=eval_config['trim_silence'])
    
    eval_loader = DataLoader(
                                eval_dataset, 
                                batch_size=eval_config['batch_size'], 
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True,
                                collate_fn=None, 
                                num_workers=8
                                )
    
    print("Number of batches in eval loader:", len(eval_loader))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    pretrained_weights_path = eval_config['pretrained_weights_path']
    
    kan_auxiliary_structure = None
    if eval_config['use_kan_auxiliary_structure']:
        kan_auxiliary_structure = ASVSpoof2019_Attr17_attack_attribute_structure
    
    model = CMMTLKAN(
                        backbone=eval_config['backbone'],
                        use_pretrained_backbone=eval_config['use_pretrained_backbone'],
                        freeze_backbone=eval_config['freeze_backbone'],
                        device=device, 
                        kan_auxiliary_structure=kan_auxiliary_structure,
                        seed=eval_config['seed']
                    ).to(device)
    
    model.load_state_dict(torch.load(pretrained_weights_path, map_location=device))
    print("Loaded pretrained model from:", pretrained_weights_path)
    
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))
    
    eval_avg_loss, eval_avg_acc, eval_balanced_acc = eval_phase(eval_loader, model, device)
    
    print("\nEvaluation Results:")
    print("\nAverage Loss:", eval_avg_loss)
    print("\nAverage Accuracy:", eval_avg_acc)
    print("\nBalanced Accuracy:", eval_balanced_acc)