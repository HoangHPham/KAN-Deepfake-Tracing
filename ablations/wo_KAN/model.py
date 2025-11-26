"""
Group-wise multi-task learning
"""

import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F

from importlib import import_module
from typing import Dict


class TaskAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, p=0.1):
        super(TaskAdapter, self).__init__()
        
        self.pre_norm = nn.LayerNorm(input_dim)
        self.down_linear = nn.Linear(input_dim, hidden_dim)
        self.up_linear = nn.Linear(hidden_dim, input_dim)
        
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p)

        self.out_layer = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        
        h = self.pre_norm(x)
        h = self.down_linear(h)
        h = self.gelu(h)
        h = self.dropout(h)
        h = self.up_linear(h)
        h = self.dropout(h)
        x = x + h # skip-connection   
        x = self.out_layer(x)
        
        return x 
    

class TaskHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TaskHead, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class CMMTL(nn.Module):
    def __init__(self, backbone='AASIST', use_pretrained_backbone=False, freeze_backbone=False, device=None):

        super(CMMTL, self).__init__()

        self.device = device
        
        self.backbone_module = None
        if backbone == 'AASIST':
            aasist_config_yaml_file = './config/backbones/AASIST.yaml'

            with open(aasist_config_yaml_file) as f:
                backbone_config = yaml.load(f, Loader=yaml.SafeLoader)

            self.backbone_module = self.get_backbone_module(backbone_config=backbone_config, 
                                                            pretrained=use_pretrained_backbone, 
                                                            freeze=freeze_backbone, 
                                                            device=device)
        elif backbone == 'SSLAASIST':
            ssl_aasist_config_yaml_file = './config/backbones/SSLAASIST.yaml'
            
            with open(ssl_aasist_config_yaml_file) as f:
                backbone_config = yaml.load(f, Loader=yaml.SafeLoader)
                
            self.backbone_module = self.get_backbone_module(backbone_config=backbone_config, 
                                                            pretrained=use_pretrained_backbone, 
                                                            freeze=freeze_backbone, 
                                                            device=device)
        
        self.task_adapters = nn.ModuleDict({
            name: TaskAdapter(input_dim=160, hidden_dim=64, output_dim=32, p=0.2)
            for name in ['T1_AS1','T2_AS2','T3_AS3','T4_AS4','T5_AS5','T6_AS6','T7_AS7']
        })
        
        self.task_heads = nn.ModuleDict({
            'T1_AS1': TaskHead(32, 3),
            'T2_AS2': TaskHead(32, 6),
            'T3_AS3': TaskHead(32, 6),
            'T4_AS4': TaskHead(32, 9),
            'T5_AS5': TaskHead(32, 5),
            'T6_AS6': TaskHead(32, 10),
            'T7_AS7': TaskHead(32, 11)
        })
        

    def get_backbone_module(self, backbone_config: Dict, pretrained: bool, freeze: bool, device: torch.device):

        module = import_module("backbones.{}".format(backbone_config["architecture"]))
        _model = getattr(module, backbone_config["architecture"])
        
        if backbone_config["architecture"] == "AASIST":
            model = _model(backbone_config).to(device, non_blocking=True) # AASIST
        elif backbone_config["architecture"] == "SSLAASIST":
            model = _model(device, use_pretrained_ssl=backbone_config['use_pretrained_ssl']).to(device, non_blocking=True) # SSLAASIST

        if pretrained:
            model.load_state_dict(torch.load(backbone_config["pretrained_weights_path"], map_location="cpu"))
            print("Pretrained weights are loaded for {} backbone".format(backbone_config["architecture"]))
        model.out_layer = nn.Identity()  # no compute, no params

        _backbone = getattr(module, f"{backbone_config['architecture']}Backbone")
        backbone = _backbone(model, freeze=freeze).to(device, non_blocking=True) # AASISTBackbone or SSLAASISTBackbone

        return backbone


    def forward(self, x):
        
        cm_embeddings = self.backbone_module(x)
        
        multi_task_outputs = {}
        
        for task_name, task_head in self.task_heads.items():
            task_adapter = self.task_adapters[task_name]
            adapted_representation = task_adapter(cm_embeddings)
            logits = task_head(adapted_representation)
            multi_task_outputs[task_name] = logits
        
        return multi_task_outputs     