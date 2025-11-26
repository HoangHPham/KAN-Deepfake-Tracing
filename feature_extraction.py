import os
import sys
import random
import numpy as np

import yaml

import warnings

from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from data_utils import fetch_protocol, Dataset_ASVspoof2019_attr17, ASVSpoof2019_Attr17_attack_attribute_structure
from data_utils import fetch_asvspoof2019_attr2_phase_bonafide, Dataset_ASVspoof2019_attr2_flacOnly
from utils import set_seed
from model import CMMTLKAN

warnings.filterwarnings("ignore", category=FutureWarning)


def extract_embeddings(loader, model, device):

    model.eval()
    
    attackAttr_embeddings = []
    attackCls_embeddings = []
    
    with torch.no_grad():
        for waveform, _ in tqdm(loader):
            
            waveform = waveform.to(device)
            
            GWMTL_outputs, AttackCls_outputs = model(waveform)
            
            multi_task_features = []
            for _, task_logits in GWMTL_outputs.items():
                multi_task_features.append(task_logits)
            
            multi_task_features = torch.cat(multi_task_features, dim=1)
            attackAttr_embeddings.append(multi_task_features.detach().cpu().numpy())
            
            attackCls_embeddings.append(AttackCls_outputs.detach().cpu().numpy())
        
    attackAttr_embeddings = np.concatenate(attackAttr_embeddings, axis=0)
    attackCls_embeddings = np.concatenate(attackCls_embeddings, axis=0)
    
    return attackAttr_embeddings, attackCls_embeddings


def extract_ASVspoof2019_attr17_phase(save_folder=None, phase='eval'):

    data_config_yaml_file = './data/ASVspoof2019_attr17_cm.yaml'
    eval_config_yaml_file = './config/evaluate.yaml'
    
    with open(data_config_yaml_file) as f:
        data_config = yaml.load(f, Loader=yaml.SafeLoader)

    with open(eval_config_yaml_file) as f:
        eval_config = yaml.load(f, Loader=yaml.SafeLoader)
    
    set_seed(eval_config)

    phase_dataset_path = data_config[f'{phase}_dataset_path']
    phase_protocols_path = data_config[f'{phase}_protocol_path']
    
    phase_flac_files, phase_ground_truths = fetch_protocol(phase_protocols_path)
    
    phase_dataset = Dataset_ASVspoof2019_attr17(
                                                flac_files=phase_flac_files, 
                                                ground_truths=phase_ground_truths, 
                                                base_dir=phase_dataset_path,
                                                phase='eval',
                                                trim_silence=eval_config['trim_silence'])
    
    phase_loader = DataLoader(
                                phase_dataset, 
                                batch_size=eval_config['batch_size'], 
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True,
                                collate_fn=None, 
                                num_workers=8
                                )
    
    print("Number of batches in phase loader:", len(phase_loader))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    pretrained_weights_path = eval_config['pretrained_weights_path']
    
    model = CMMTLKAN(
                        backbone=eval_config['backbone'],
                        use_pretrained_backbone=eval_config['use_pretrained_backbone'],
                        freeze_backbone=eval_config['freeze_backbone'],
                        device=device, 
                        kan_auxiliary_structure=ASVSpoof2019_Attr17_attack_attribute_structure,
                        seed=eval_config['seed']
                    ).to(device)
    
    model.load_state_dict(torch.load(pretrained_weights_path, map_location=device))
    print("Loaded pretrained model from:", pretrained_weights_path)
    
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))
    
    attackAttr_embds, attackCls_embds =  extract_embeddings(phase_loader, model, device)
    print("AttackAttr embeddings shape:", attackAttr_embds.shape)
    print("AttackCls embeddings shape:", attackCls_embds.shape)
    
    attAttr_embds_save_path = f"{save_folder}/ASVspoof2019_attr17_{phase}_attackAttr_embeddings.npy"
    attCls_embds_save_path = f"{save_folder}/ASVspoof2019_attr17_{phase}_attackCls_embeddings.npy"
    
    np.save(attAttr_embds_save_path, attackAttr_embds)
    np.save(attCls_embds_save_path, attackCls_embds)
    
    print(f"Saved {phase} attackAttr embeddings to:", attAttr_embds_save_path)
    print(f"Saved {phase} attackCls embeddings to:", attCls_embds_save_path)


def extract_ASVspoof2019_attr2_full_bonafide(save_folder=None):
    
    data_config_yaml_file = './data/ASVspoof2019_LA_cm.yaml'
    eval_config_yaml_file = './config/evaluate.yaml'
    
    with open(data_config_yaml_file) as f:
        data_config = yaml.load(f, Loader=yaml.SafeLoader)
    
    with open(eval_config_yaml_file) as f:
        eval_config = yaml.load(f, Loader=yaml.SafeLoader)
    
    asvspoof2019_attr2_train_protocols_path = data_config["train_protocol_path"] 
    asvspoof2019_attr2_dev_protocols_path = data_config["dev_protocol_path"] 
    asvspoof2019_attr2_eval_protocols_path = data_config["eval_protocol_path"] 
    
    asvspoof2019_attr2_train_dataset_path = data_config["train_dataset_path"]
    asvspoof2019_attr2_dev_dataset_path = data_config["dev_dataset_path"]
    asvspoof2019_attr2_eval_dataset_path = data_config["eval_dataset_path"]
    
    train_flac_files = fetch_asvspoof2019_attr2_phase_bonafide(asvspoof2019_attr2_train_protocols_path)
    dev_flac_files = fetch_asvspoof2019_attr2_phase_bonafide(asvspoof2019_attr2_dev_protocols_path)
    eval_flac_files = fetch_asvspoof2019_attr2_phase_bonafide(asvspoof2019_attr2_eval_protocols_path)
    
    asvspoof2019_attr2_train_bonafide_dataset = Dataset_ASVspoof2019_attr2_flacOnly(
                                                                                    flac_files=train_flac_files, 
                                                                                    base_dir=asvspoof2019_attr2_train_dataset_path,
                                                                                    trim_silence=eval_config['trim_silence']
                                                                                    )
    
    asvspoof2019_attr2_dev_bonafide_dataset = Dataset_ASVspoof2019_attr2_flacOnly(
                                                                                    flac_files=dev_flac_files, 
                                                                                    base_dir=asvspoof2019_attr2_dev_dataset_path,
                                                                                    trim_silence=eval_config['trim_silence']
                                                                                    )
    
    asvspoof2019_attr2_eval_bonafide_dataset = Dataset_ASVspoof2019_attr2_flacOnly(
                                                                                    flac_files=eval_flac_files, 
                                                                                    base_dir=asvspoof2019_attr2_eval_dataset_path,
                                                                                    trim_silence=eval_config['trim_silence']
                                                                                    )
    
    asvspoof2019_attr2_train_bonafide_loader = DataLoader(
                                                            asvspoof2019_attr2_train_bonafide_dataset, 
                                                            batch_size=eval_config['batch_size'], 
                                                            shuffle=False,
                                                            drop_last=False,
                                                            pin_memory=True,
                                                            collate_fn=None, 
                                                            num_workers=8
                                                            )
    
    asvspoof2019_attr2_dev_bonafide_loader = DataLoader(
                                                            asvspoof2019_attr2_dev_bonafide_dataset, 
                                                            batch_size=eval_config['batch_size'], 
                                                            shuffle=False,
                                                            drop_last=False,
                                                            pin_memory=True,
                                                            collate_fn=None, 
                                                            num_workers=8
                                                            )
    
    asvspoof2019_attr2_eval_bonafide_loader = DataLoader(
                                                            asvspoof2019_attr2_eval_bonafide_dataset, 
                                                            batch_size=eval_config['batch_size'], 
                                                            shuffle=False,
                                                            drop_last=False,
                                                            pin_memory=True,
                                                            collate_fn=None, 
                                                            num_workers=8
                                                            )
    
    print("Number of batches in train loader:", len(asvspoof2019_attr2_train_bonafide_loader))
    print("Number of batches in dev loader:", len(asvspoof2019_attr2_dev_bonafide_loader))
    print("Number of batches in eval loader:", len(asvspoof2019_attr2_eval_bonafide_loader))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    pretrained_weights_path = eval_config['pretrained_weights_path']
    
    model = CMMTLKAN(
                        backbone=eval_config['backbone'],
                        use_pretrained_backbone=eval_config['use_pretrained_backbone'],
                        freeze_backbone=eval_config['freeze_backbone'],
                        device=device, 
                        kan_auxiliary_structure=ASVSpoof2019_Attr17_attack_attribute_structure,
                        seed=eval_config['seed']
                    ).to(device)
    
    model.load_state_dict(torch.load(pretrained_weights_path, map_location=device))
    print("Loaded pretrained model from:", pretrained_weights_path)
    
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))
    
    train_attackAttr_embds, train_attackCls_embds =  extract_embeddings(asvspoof2019_attr2_train_bonafide_loader, model, device)
    dev_attackAttr_embds, dev_attackCls_embds =  extract_embeddings(asvspoof2019_attr2_dev_bonafide_loader, model, device)
    eval_attackAttr_embds, eval_attackCls_embds =  extract_embeddings(asvspoof2019_attr2_eval_bonafide_loader, model, device)
    
    print("\nAttackAttr embeddings of ASVspoof2019_attr2 train phase bonafide has shape:", train_attackAttr_embds.shape)
    print("AttackCls embeddings of ASVspoof2019_attr2 train phase bonafide has shape:", train_attackCls_embds.shape)
    print("\nAttackAttr embeddings of ASVspoof2019_attr2 dev phase bonafide has shape:", dev_attackAttr_embds.shape)
    print("AttackCls embeddings of ASVspoof2019_attr2 dev phase bonafide has shape:", dev_attackCls_embds.shape)
    print("\nAttackAttr embeddings of ASVspoof2019_attr2 eval phase bonafide has shape:", eval_attackAttr_embds.shape)
    print("AttackCls embeddings of ASVspoof2019_attr2 eval phase bonafide has shape:", eval_attackCls_embds.shape)
    
    ASVspoof2019_attr2_full_bonafide_attackAttr_embds = [train_attackAttr_embds, dev_attackAttr_embds, eval_attackAttr_embds]
    ASVspoof2019_attr2_full_bonafide_attackAttr_embds = np.concatenate(ASVspoof2019_attr2_full_bonafide_attackAttr_embds, axis=0)
    print("\nFull shape of AttackAttr ASVspoof2019_attr2 bonafide: ", ASVspoof2019_attr2_full_bonafide_attackAttr_embds.shape)
    
    ASVspoof2019_attr2_full_bonafide_attackCls_embds = [train_attackCls_embds, dev_attackCls_embds, eval_attackCls_embds]
    ASVspoof2019_attr2_full_bonafide_attackCls_embds = np.concatenate(ASVspoof2019_attr2_full_bonafide_attackCls_embds, axis=0)
    print("Full shape of AttackCls ASVspoof2019_attr2 bonafide: ", ASVspoof2019_attr2_full_bonafide_attackCls_embds.shape)
    
    attAttr_embds_save_path = f"{save_folder}/ASVspoof2019_attr2_full_bonafide_attackAttr_embeddings.npy"
    attCls_embds_save_path = f"{save_folder}/ASVspoof2019_attr2_full_bonafide_attackCls_embeddings.npy"
    
    np.save(attAttr_embds_save_path, ASVspoof2019_attr2_full_bonafide_attackAttr_embds)
    np.save(attCls_embds_save_path, ASVspoof2019_attr2_full_bonafide_attackCls_embds)
    
    print(f"\nSaved attackAttr full bonafide embeddings to:", attAttr_embds_save_path)
    print(f"Saved attackCls full bonafide embeddings to:", attCls_embds_save_path)
    

if __name__ == "__main__":
    
    exp_name = 'ST_RB_AASIST_MTL_KANaux_trainingScratch_seed42_bs64'
    
    save_folder = os.path.join('./extracted_embds', exp_name)
    os.makedirs(save_folder, exist_ok=True)
    
    # extract_ASVspoof2019_attr17_phase(save_folder=save_folder, phase='train')
    # extract_ASVspoof2019_attr17_phase(save_folder=save_folder, phase='dev')
    extract_ASVspoof2019_attr17_phase(save_folder=save_folder, phase='eval')
    
    
    # extract_ASVspoof2019_attr2_full_bonafide(save_folder=save_folder)
