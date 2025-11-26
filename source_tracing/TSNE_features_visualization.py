import os
import sys
import random
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import yaml

import warnings

from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from data_utils import fetch_protocol, Dataset_ASVspoof2019_attr17, ASVSpoof2019_Attr17_attack_attribute_structure
from data_utils import fetch_asvspoof2019_attr2_phase_bonafide, Dataset_ASVspoof2019_attr2_flacOnly, collate_fn, collate_fn_bonafide
from utils import set_seed
from model import CMMTLKAN

warnings.filterwarnings("ignore", category=FutureWarning)


def extract_attackAttr_embeddings(loader, model, device):

    model.eval()
    
    attackAttr_embeddings = []
    
    with torch.no_grad():
        for waveform, _ in tqdm(loader):
            
            waveform = waveform.to(device)
            
            GWMTL_outputs, _ = model(waveform)
            
            multi_task_features = []
            
            for _, task_logits in GWMTL_outputs.items():
                
                multi_task_features.append(task_logits)
            
            multi_task_features = torch.cat(multi_task_features, dim=1)
            
            attackAttr_embeddings.append(multi_task_features.detach().cpu().numpy())
        
        attackAttr_embeddings = np.concatenate(attackAttr_embeddings, axis=0)
    
    return attackAttr_embeddings


def extract_ASVspoof2019_attr17_phase(save_path=None, phase='eval'):

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
                                                ST=eval_config['silence_trimming'])
    
    phase_loader = DataLoader(
                                phase_dataset, 
                                batch_size=eval_config['batch_size'], 
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True,
                                collate_fn=collate_fn, 
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
    
    attackAttr_embds =  extract_attackAttr_embeddings(phase_loader, model, device)
    print("AttackAttr embeddings shape:", attackAttr_embds.shape)
    
    np.save(save_path, attackAttr_embds)
    print("Saved attackAttr embeddings to:", save_path)


def extract_ASVspoof2019_attr2_full_bonafide(save_path=None):
    
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
                                                                                    ST=eval_config['silence_trimming']
                                                                                    )
    
    asvspoof2019_attr2_dev_bonafide_dataset = Dataset_ASVspoof2019_attr2_flacOnly(
                                                                                    flac_files=dev_flac_files, 
                                                                                    base_dir=asvspoof2019_attr2_dev_dataset_path,
                                                                                    ST=eval_config['silence_trimming']
                                                                                    )
    
    asvspoof2019_attr2_eval_bonafide_dataset = Dataset_ASVspoof2019_attr2_flacOnly(
                                                                                    flac_files=eval_flac_files, 
                                                                                    base_dir=asvspoof2019_attr2_eval_dataset_path,
                                                                                    ST=eval_config['silence_trimming']
                                                                                    )
    
    asvspoof2019_attr2_train_bonafide_loader = DataLoader(
                                                            asvspoof2019_attr2_train_bonafide_dataset, 
                                                            batch_size=eval_config['batch_size'], 
                                                            shuffle=False,
                                                            drop_last=False,
                                                            pin_memory=True,
                                                            collate_fn=collate_fn_bonafide, 
                                                            num_workers=8
                                                            )
    
    asvspoof2019_attr2_dev_bonafide_loader = DataLoader(
                                                            asvspoof2019_attr2_dev_bonafide_dataset, 
                                                            batch_size=eval_config['batch_size'], 
                                                            shuffle=False,
                                                            drop_last=False,
                                                            pin_memory=True,
                                                            collate_fn=collate_fn_bonafide, 
                                                            num_workers=8
                                                            )
    
    asvspoof2019_attr2_eval_bonafide_loader = DataLoader(
                                                            asvspoof2019_attr2_eval_bonafide_dataset, 
                                                            batch_size=eval_config['batch_size'], 
                                                            shuffle=False,
                                                            drop_last=False,
                                                            pin_memory=True,
                                                            collate_fn=collate_fn_bonafide, 
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
    
    train_attackAttr_embds =  extract_attackAttr_embeddings(asvspoof2019_attr2_train_bonafide_loader, model, device)
    print("AttackCls embeddings of ASVspoof2019_attr2 train phase bonafide has shape:", train_attackAttr_embds.shape)
    
    dev_attackAttr_embds =  extract_attackAttr_embeddings(asvspoof2019_attr2_dev_bonafide_loader, model, device)
    print("AttackCls embeddings of ASVspoof2019_attr2 dev phase bonafide has shape:", dev_attackAttr_embds.shape)
    
    eval_attackAttr_embds =  extract_attackAttr_embeddings(asvspoof2019_attr2_eval_bonafide_loader, model, device)
    print("AttackCls embeddings of ASVspoof2019_attr2 eval phase bonafide has shape:", eval_attackAttr_embds.shape)
    
    ASVspoof2019_attr2_full_bonafide_attackAttr_embds = [train_attackAttr_embds, dev_attackAttr_embds, eval_attackAttr_embds]
    ASVspoof2019_attr2_full_bonafide_attackAttr_embds = np.concatenate(ASVspoof2019_attr2_full_bonafide_attackAttr_embds, axis=0)
    
    print("Full shape: ", ASVspoof2019_attr2_full_bonafide_attackAttr_embds.shape)
    
    np.save(save_path, ASVspoof2019_attr2_full_bonafide_attackAttr_embds)
    print("Saved attackCls embeddings to:", save_path)


def tsna_visualization(spoof_embeds_path=None,
                       bonafide_embeds_path=None,
                       save_fig_path=None,
                       title=None):
    
    # --- Inputs you already have ---
    # X_attack: (47880, 50) attack features
    # y_attack: (47880,) integer labels in {0,...,16} for 17 attack types
    # X_bona:   (7355, 50)  bonafide features (single class)

    
    X_attack = np.load(spoof_embeds_path)
    
    data_config_yaml_file = './data/ASVspoof2019_attr17_cm.yaml'
    
    with open(data_config_yaml_file) as f:
        data_config = yaml.load(f, Loader=yaml.SafeLoader)
    
    eval_protocols_path = data_config['train_protocol_path']
    _, eval_ground_truths = fetch_protocol(eval_protocols_path)
    
    y_attack = []
    
    for gt in eval_ground_truths:
        y_attack.append(gt['Attack_id'][0])
    y_attack = np.array(y_attack, dtype=int)
    
    X_bona = np.load(bonafide_embeds_path)
    
    print("Attack features shape:", X_attack.shape)
    print("Bonafide features shape:", X_bona.shape)
    
    # --------- Optional: Downsample for speed (toggle here) ----------
    enable_downsample = False
    per_class_max = 3000   # cap per attack type
    bona_max = 3000        # cap bonafide
    if enable_downsample:
        idx_keep = []
        for c in np.unique(y_attack):
            idx_c = np.where(y_attack == c)[0]
            if len(idx_c) > per_class_max:
                idx_c = np.random.RandomState(0).choice(idx_c, size=per_class_max, replace=False)
            idx_keep.append(idx_c)
        idx_keep = np.concatenate(idx_keep)
        X_attack_ds = X_attack[idx_keep]
        y_attack_ds = y_attack[idx_keep]

        if len(X_bona) > bona_max:
            bona_idx = np.random.RandomState(0).choice(len(X_bona), size=bona_max, replace=False)
            X_bona_ds = X_bona[bona_idx]
        else:
            X_bona_ds = X_bona
    else:
        X_attack_ds, y_attack_ds = X_attack, y_attack
        X_bona_ds = X_bona

    # ------------- Combine & scale -------------
    X = np.vstack([X_attack_ds, X_bona_ds])
    # create labels: attack classes 0..16, bonafide as 17
    y = np.concatenate([y_attack_ds, np.full((len(X_bona_ds),), 17, dtype=int)])

    X = StandardScaler().fit_transform(X)

    # ------------- (Optional) PCA pre-processing -------------
    # t-SNE often benefits from PCA to ~30-50 dims
    pca = PCA(n_components=min(30, X.shape[1]), random_state=42)
    X_pca = pca.fit_transform(X)

    # ------------- t-SNE -------------
    tsne = TSNE(
        n_components=2,
        learning_rate='auto',
        init='pca',
        perplexity=50,        # works well for a few thousand+ points; adjust if you downsample a lot
        n_iter=1000,
        random_state=42,
        verbose=1
    )
    X_2d = tsne.fit_transform(X_pca)

    # ------------- Plotting -------------
    fig, ax = plt.subplots(figsize=(9, 7), dpi=120)

    # Colors: 17 distinct colors + grey for bonafide
    # Use tab20 (20 distinct colors); take first 17 for attacks.
    tab20 = plt.get_cmap('tab20', 20)
    attack_colors = [tab20(i) for i in range(17)]
    bona_color = (1.0, 1.0, 0.0, 0.8)  # grey with some transparency

    # Plot bonafide first (underlay), then attacks
    mask_bona = (y == 17)
    ax.scatter(
        X_2d[mask_bona, 0], X_2d[mask_bona, 1],
        s=5, c=[bona_color], label='bonafide', alpha=0.6, linewidths=0
    )

    # Plot each attack class
    
    attack_types = ["A01", "A02", "A03", "A04-16", "A05", "A06-19", "A07", "A08", "A09", "A10", "A11", "A12", "A13", "A14", "A15", "A17", "A18"]
    
    for c in range(17):
        m = (y == c)
        if np.any(m):
            ax.scatter(
                X_2d[m, 0], X_2d[m, 1],
                s=5, c=[attack_colors[c]],
                label=attack_types[c], alpha=0.9, linewidths=0
            )

    # Cosmetics
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('t-SNE dimension 1')
    ax.set_ylabel('t-SNE dimension 2')
    ax.set_title(title)

    # Manage legend (many entries) – place outside
    leg = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), title='Classes', fontsize=8)
    for lh in leg.legend_handles:  # smaller dots in legend
        lh.set_sizes([30])

    plt.tight_layout()
    plt.savefig(save_fig_path, dpi=200)
    plt.show()

    print("Saved t-SNE figure to:", save_fig_path)


if __name__ == "__main__":
    
    save_folder = './AttackAttr_embds'
    os.makedirs(save_folder, exist_ok=True)
    
    ### Extract attackAttr embeddings for ASVspoof2019_attr17_{phase}
    # save_path = os.path.join(save_folder, 'ASVspoof2019_attr17_eval_attackAttr_embeddings_ST_AASIST_MTL_KANaux_partialFinetuning.npy')
    # extract_ASVspoof2019_attr17_phase(save_path, phase='eval')
    
    ### Extract attackAttr embeddings for ASVspoof2019_attr2_full_bonafide
    # save_path = os.path.join(save_folder, 'ASVspoof2019_attr2_full_bonafide_attackAttr_embeddings_ST_AASIST_MTL_KANaux_partialFinetuning.npy')
    # extract_ASVspoof2019_attr2_full_bonafide(save_path)
    
    ### t-SNE visualization of attackAttr embeddings
    ASVspoof2019_attr17_eval_attackAttr_embds_path = os.path.join(save_folder, 'ASVspoof2019_attr17_train_attackAttr_embeddings_AASIST_MTL_KANaux_partialFinetuning.npy')
    ASVspoof2019_attr2_full_bonafide_attackAttr_embds_path = os.path.join(save_folder, 'ASVspoof2019_attr2_full_bonafide_attackAttr_embeddings_AASIST_MTL_KANaux_partialFinetuning.npy')
    save_fig_path = os.path.join(save_folder, 'TSNE_ASVspoof2019_attr17_eval_vs_attr2_full_bonafide_AASIST_MTL_KANaux_partialFinetuning.png')
    title = "t-SNE of 17 attack types vs bonafide"
    tsna_visualization(ASVspoof2019_attr17_eval_attackAttr_embds_path,
                       ASVspoof2019_attr2_full_bonafide_attackAttr_embds_path,
                       save_fig_path,
                       title=title)
    
    
    ### t-SNE visualization of attackCls embeddings
    # attackCls_embds_folder = './AttackCls_embds'
    
    # ASVspoof2019_attr17_eval_attackCls_embds_path = os.path.join(attackCls_embds_folder, 'ASVspoof2019_attr17_eval_attackCls_embeddings_AASIST_MTL_KANaux_trainingScratch.npy')
    # ASVspoof2019_attr2_full_bonafide_attackCls_embds_path = os.path.join(attackCls_embds_folder, 'ASVspoof2019_attr2_full_bonafide_attackCls_embeddings_AASIST_MTL_KANaux_trainingScratch.npy')
    # save_fig_path = os.path.join(attackCls_embds_folder, 'TSNE_ASVspoof2019_attr17_eval_vs_attr2_full_bonafide_AASIST_MTL_KANaux_trainingScratch.png')
    # title = "t-SNE of 17 attack types vs bonafide"
    # tsna_visualization(ASVspoof2019_attr17_eval_attackCls_embds_path,
    #                    ASVspoof2019_attr2_full_bonafide_attackCls_embds_path,
    #                    save_fig_path,
    #                    title=title)
    
    