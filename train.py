
import sys
import random
import argparse
import numpy as np

import matplotlib.pyplot as plt
import pickle
import yaml

import warnings

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from data_utils import fetch_protocol, Dataset_ASVspoof2019_attr17, ASVSpoof2019_Attr17_attack_attribute_structure
from utils import seed_worker, set_seed
from model import CMMTLKAN

warnings.filterwarnings("ignore", category=FutureWarning)


def main(args: argparse.Namespace) -> None:
    
    data_config_yaml_file = './data/ASVspoof2019_attr17_cm.yaml'
    train_config_yaml_file = './config/train.yaml'

    with open(data_config_yaml_file) as f:
        data_config = yaml.load(f, Loader=yaml.SafeLoader)

    with open(train_config_yaml_file) as f:
        train_config = yaml.load(f, Loader=yaml.SafeLoader)

    # make experiment reproducible
    set_seed(train_config)
    gen = torch.Generator()
    gen.manual_seed(train_config['seed'])

    train_dataset_path = data_config['train_dataset_path']
    dev_dataset_path = data_config['dev_dataset_path']
    
    train_protocols_path = data_config['train_protocol_path']
    dev_protocols_path = data_config['dev_protocol_path']
    
    train_flac_files, train_ground_truths = fetch_protocol(train_protocols_path)
    dev_flac_files, dev_ground_truths = fetch_protocol(dev_protocols_path)

    train_dataset = Dataset_ASVspoof2019_attr17(
                                                flac_files=train_flac_files, 
                                                ground_truths=train_ground_truths, 
                                                base_dir=train_dataset_path,
                                                phase='train',
                                                trim_silence=train_config['trim_silence'],
                                                rawboost_args=args
                                                )

    dev_dataset = Dataset_ASVspoof2019_attr17(
                                                flac_files=dev_flac_files, 
                                                ground_truths=dev_ground_truths, 
                                                base_dir=dev_dataset_path,
                                                phase='dev',
                                                trim_silence=train_config['trim_silence']
                                                )
    
    train_loader = DataLoader(
                                train_dataset, 
                                batch_size=train_config['batch_size'], 
                                shuffle=True,
                                drop_last=True,
                                pin_memory=True,
                                worker_init_fn=seed_worker,
                                generator=gen,
                                collate_fn=None, 
                                num_workers=8
                                )

    dev_loader = DataLoader(
                                dev_dataset, 
                                batch_size=train_config['batch_size'], 
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True,
                                collate_fn=None, 
                                num_workers=8
                                )
    
    print("Number of batches in train loader:", len(train_loader))
    print("Number of batches in dev loader:", len(dev_loader))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    kan_auxiliary_structure = None
    if train_config['use_kan_auxiliary_structure']:
        kan_auxiliary_structure = ASVSpoof2019_Attr17_attack_attribute_structure

    model = CMMTLKAN(
                        backbone=train_config['backbone'],
                        use_pretrained_backbone=train_config['use_pretrained_backbone'],
                        freeze_backbone=train_config['freeze_backbone'],
                        device=device, 
                        kan_auxiliary_structure=kan_auxiliary_structure,
                        seed=train_config['seed']
                    ).to(device)

    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))
    
    print("Backbone architecture: ", train_config['backbone'])
    print("use_pretrained_backbone:", train_config['use_pretrained_backbone'])
    print("freeze_backbone:", train_config['freeze_backbone'])
    
    if train_config['resume']:
        assert train_config['pretrained_ppm_path'] is not None, "pretrained_ppm_path should not be None if resume is True"
        model.load_state_dict(torch.load(train_config['pretrained_ppm_path'], map_location=device))
        print("Resumed weights from:", train_config['pretrained_ppm_path'])

    optimizer = torch.optim.Adam(
                                    model.parameters(),
                                    lr=train_config['learning_rate'],
                                    betas=train_config['betas'],
                                    weight_decay=train_config['weight_decay'],
                                    amsgrad=train_config['amsgrad'])

    exp_name = train_config['exp_name']
    print("Experiment name: ", exp_name)

    n_epochs = train_config['n_epochs']

    history = {
        'train_loss': [],
        'train_acc': [],
        'dev_loss': [],
        'dev_acc': []
    }
    
    best_acc = -1e9
    
    for epoch in range(n_epochs):
        print("\n====> training epoch {:03d} / {:03d}".format(epoch+1, n_epochs))
        
        train_avg_loss, train_avg_acc = train_phase(train_loader, model, train_config['freeze_backbone'], optimizer, device)
        history['train_loss'].append(train_avg_loss)
        history['train_acc'].append(train_avg_acc)

        print("Training loss: {} - Training accuracy: {}".format(train_avg_loss, train_avg_acc))
        
        dev_avg_loss, dev_avg_acc = dev_phase(dev_loader, model, device)
        history['dev_loss'].append(dev_avg_loss)
        history['dev_acc'].append(dev_avg_acc)
        
        print("Dev loss: {} - Dev accuracy: {}".format(dev_avg_loss, dev_avg_acc))
        
        torch.save(model.state_dict(), f'weights/last_{exp_name}.pt')
        
        if dev_avg_acc['total_acc'] > best_acc:
            best_acc = dev_avg_acc['total_acc']
            torch.save(model.state_dict(), f'weights/best_{exp_name}.pt')
            
    plot(history, exp_name)
    
    # save history
    with open(f"logs/history_{exp_name}.pkl", "wb") as f:
        pickle.dump(history, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Training completed.")


def plot(history, exp_name):
    """
    history: dict with keys 'train_loss', 'train_acc', 'dev_loss', 'dev_acc'
    in which:
    - 'train_loss' and 'dev_loss' are lists of loss (dict) for each epoch
    - 'train_acc' and 'dev_acc' are lists of accuracy (dict) for each epoch
    """
    
    tasks = list(history['train_loss'][0].keys())  # Get task names dynamically
    epochs = range(1, len(history['train_loss']) + 1)

    # -------- Plot LOSS --------
    fig_loss, axes_loss = plt.subplots(nrows=4, ncols=3, figsize=(30, 15))
    axes_loss = axes_loss.flatten()

    for i, task in enumerate(tasks):
        train_vals = [epoch_dict[task] for epoch_dict in history['train_loss']]
        dev_vals   = [epoch_dict[task] for epoch_dict in history['dev_loss']]

        ax = axes_loss[i]
        ax.plot(epochs, train_vals, label='Train Loss')
        ax.plot(epochs, dev_vals, label='Dev Loss')
        ax.set_title(f'{task}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)

    fig_loss.tight_layout()
    plt.savefig(f"plots/loss_plot_{exp_name}.png")
    plt.show()

    tasks = list(history['train_acc'][0].keys())  # Get task names dynamically
    
    # -------- Plot ACCURACY --------
    fig_acc, axes_acc = plt.subplots(nrows=4, ncols=3, figsize=(30, 15))
    axes_acc = axes_acc.flatten()

    for i, task in enumerate(tasks):
        train_vals = [epoch_dict[task] for epoch_dict in history['train_acc']]
        dev_vals   = [epoch_dict[task] for epoch_dict in history['dev_acc']]

        ax = axes_acc[i]
        ax.plot(epochs, train_vals, label='Train Acc')
        ax.plot(epochs, dev_vals, label='Dev Acc')
        ax.set_title(f'{task}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True)

    fig_acc.tight_layout()
    plt.savefig(f"plots/accuracy_plot_{exp_name}.png")
    plt.show()
    
    
def train_phase(train_loader=None, model=None, freeze_backbone=False, optimizer=None, device=None):
    
    model.train()

    if freeze_backbone:
        model.backbone_module.eval()
    
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
    
    total_spoof_samples = 0
    
    for waveform, ground_truth in train_loader:
        
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
        
        running_loss = multi_task_losses.mean() + AttackCls_loss.mean() # total loss for 1 batch
        
        total_loss['running_loss'] += running_loss.item()  # accumulate total loss
        
        optimizer.zero_grad()
        running_loss.backward()
        optimizer.step()
        
    avg_loss = {
        **{task_name: loss_value / total_spoof_samples for task_name, loss_value in total_loss.items() if task_name != "running_loss" and task_name != "AttackCls_loss"}
    }
    avg_loss['AttackCls_loss'] = total_loss['AttackCls_loss'] / total_spoof_samples
    avg_loss["running_loss"] = total_loss["running_loss"] / len(train_loader)
    
    avg_acc = {
        **{task_name: acc / total_spoof_samples for task_name, acc in total_acc.items() if task_name != "AttackCls_acc" and task_name != "AttackAttr_acc"}
    }
    avg_acc["AttackAttr_acc"] = total_acc['AttackAttr_acc'] / (total_spoof_samples * 7)
    avg_acc["AttackCls_acc"] = total_acc['AttackCls_acc'] / total_spoof_samples
    avg_acc["total_acc"] = (avg_acc["AttackAttr_acc"] + avg_acc["AttackCls_acc"]) / 2
    
    return avg_loss, avg_acc


def dev_phase(loader, model, device):

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
    
    total_spoof_samples = 0
    
    with torch.no_grad():
        for waveform, ground_truth in loader:
            
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
    
    return avg_loss, avg_acc


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="ASVspoof2019-attr17 source tracing training")
    
    ##===================================================Rawboost data augmentation parameters======================================================================#

    parser.add_argument('--algo', type=int, default=0, 
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')

    # LnL_convolutive_noise parameters 
    parser.add_argument('--nBands', type=int, default=5, 
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000, 
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000, 
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10, 
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, 
                    help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, 
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')
    
    ##===================================================Rawboost data augmentation ======================================================================#
    
    main(parser.parse_args())