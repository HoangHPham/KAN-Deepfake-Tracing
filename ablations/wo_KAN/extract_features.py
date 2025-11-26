import os
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
from model import CMMTL

warnings.filterwarnings("ignore", category=FutureWarning)


def extract_embeddings(loader, model, device):

    model.eval()
    
    attackAttr_embeddings = []
    
    with torch.no_grad():
        for waveform, _ in tqdm(loader):
            
            waveform = waveform.to(device)
            
            GWMTL_outputs = model(waveform)
            
            multi_task_features = []
            for _, task_logits in GWMTL_outputs.items():
                multi_task_features.append(task_logits)
            
            multi_task_features = torch.cat(multi_task_features, dim=1)
            attackAttr_embeddings.append(multi_task_features.detach().cpu().numpy())
            
    attackAttr_embeddings = np.concatenate(attackAttr_embeddings, axis=0)
    
    return attackAttr_embeddings


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
    
    model = CMMTL(
                    backbone=eval_config['backbone'],
                    use_pretrained_backbone=eval_config['use_pretrained_backbone'],
                    freeze_backbone=eval_config['freeze_backbone'],
                    device=device, 
                ).to(device)
    
    model.load_state_dict(torch.load(pretrained_weights_path, map_location=device))
    print("Loaded pretrained model from:", pretrained_weights_path)
    
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))
    
    attackAttr_embds =  extract_embeddings(phase_loader, model, device)
    print("AttackAttr embeddings shape:", attackAttr_embds.shape)
    
    attAttr_embds_save_path = f"{save_folder}/ASVspoof2019_attr17_{phase}_AblateKAN_attackAttr_embeddings.npy"
    
    np.save(attAttr_embds_save_path, attackAttr_embds)
    
    print(f"Saved {phase} attackAttr embeddings to:", attAttr_embds_save_path)



if __name__ == "__main__":
    
    exp_name = 'ST_RB_AASIST_MTL_pf'
    
    save_folder = os.path.join('./extracted_embds', exp_name)
    os.makedirs(save_folder, exist_ok=True)
    
    # extract_ASVspoof2019_attr17_phase(save_folder=save_folder, phase='train')
    # extract_ASVspoof2019_attr17_phase(save_folder=save_folder, phase='dev')
    extract_ASVspoof2019_attr17_phase(save_folder=save_folder, phase='eval')