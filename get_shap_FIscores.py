import os
import shap
import random
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from data_utils import ASVSpoof2019_Attr17_attack_attribute_structure
from model import CMMTLKAN

from tqdm.auto import tqdm

import pickle


def set_seed(seed: int = 42):
    random.seed(seed)                    
    np.random.seed(seed)                 
    torch.manual_seed(seed)               
    torch.cuda.manual_seed(seed)          
    torch.cuda.manual_seed_all(seed)      

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def GWMTL_extract_features(model, loader, device):
    
    model.eval()
    
    with torch.no_grad():
        
        all_features = []
        
        for cm_embeddings, _ in loader:
        
            cm_embeddings = cm_embeddings.to(device)

            GWMTL_outputs, AttackCls_outputs = model(cm_embeddings)
            
            multi_task_features = []
            
            for task_name, task_logits in GWMTL_outputs.items():
                
                multi_task_features.append(task_logits.cpu().numpy())
                
            multi_task_features = np.concatenate(multi_task_features, axis=1)
            
            all_features.append(multi_task_features)
            
        all_features = np.concatenate(all_features, axis=0)
        
        return all_features


class KANOnlyWrapper(nn.Module):
    """
    Bọc riêng phần KAN. Đầu vào: (B, 50) – chính là multi_task_features.
    Đầu ra: (B, 17). Tuỳ cách bạn huấn luyện, có thể để logits hoặc softmax.
    """
    def __init__(self, kan_module: nn.Module, apply_softmax: bool = False):
        super().__init__()
        self.kan = kan_module
        self.apply_softmax = apply_softmax

    def forward(self, x_kan):
        out = self.kan(x_kan)  # (B, 17)
        if self.apply_softmax:
            out = torch.softmax(out, dim=1)
        return out
    

if __name__ == "__main__":   
    
    seed = 42
    set_seed(seed)
    
    print("get Shap FI scores for GWMTL-KAN aux: seed = 4321, bg_data with 5000 samples")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    exp_name = 'ST_RB_SSLAASIST_MTL_KANaux_fullFinetuning'
    embds_folder = os.path.join('./extracted_embds', exp_name)
    assert os.path.exists(embds_folder), f"Folder {embds_folder} does not exist."
    ASVspoof2019_attr17_eval_attackAttr_embds_path = os.path.join(embds_folder, 'ASVspoof2019_attr17_eval_attackAttr_embeddings.npy')
    kan_input_features = np.load(ASVspoof2019_attr17_eval_attackAttr_embds_path)
    
    pretrained_weights_path = "./weights/best_ST_RB_AASIST_MTL_KANaux_trainingScratch.pt"
    model = CMMTLKAN(
                        backbone='AASIST',
                        use_pretrained_backbone=False,
                        freeze_backbone=False,
                        device=device, 
                        kan_auxiliary_structure=ASVSpoof2019_Attr17_attack_attribute_structure,
                        seed=seed
                    ).to(device)
    model.load_state_dict(torch.load(pretrained_weights_path, map_location=device))
    print("Loaded pretrained model from:", pretrained_weights_path)
    
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))
    
    kan_module = model.kan_module
    
    # Background: tóm gọn bằng k-means (ví dụ k=64)
    random_indexs = np.random.choice(kan_input_features.shape[0], 5000, replace=False)
    background = shap.kmeans(kan_input_features[random_indexs], k=64)

    # Bọc KAN
    kan_only = KANOnlyWrapper(kan_module, apply_softmax=False).to(device).eval()

    # DeepExplainer cần torch.Tensor làm background
    bg_t = torch.tensor(background.data, dtype=torch.float32, device=device)

    explainer = shap.DeepExplainer(kan_only, bg_t)
    
    # Eval samples: chọn all mẫu để tính SHAP (tuỳ tài nguyên)
    total_shap_values = []
    N = kan_input_features.shape[0]
    for i in tqdm(range(N)):
        X_eval = kan_input_features[i:i+1, :]
        X_eval_t = torch.tensor(X_eval, dtype=torch.float32, device=device)
        shap_values = explainer.shap_values(X_eval_t, check_additivity=False)
        shap_values = np.squeeze(shap_values, axis=0)
        total_shap_values.append(shap_values)
    total_shap_values = np.array(total_shap_values)

    save_folder = "./shap_values"
    os.makedirs(save_folder, exist_ok=True)
    save_file_path = os.path.join(save_folder, "shap_values_AASIST_MTL_KANaux.pkl")
    
    pickle.dump(total_shap_values, open(save_file_path, "wb"))
    print("DONE")