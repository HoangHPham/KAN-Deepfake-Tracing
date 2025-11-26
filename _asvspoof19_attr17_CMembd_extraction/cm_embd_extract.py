"""
Script that loads the AASIST model, generates 160-dimensional embeddings for train/dev/eval set, and saves it.

"""

# importing libraries
import os
import json
import warnings
import numpy as np

from tqdm import tqdm
from pathlib import Path
from typing import Dict, List
from importlib import import_module

import torch
from torch.utils.data import DataLoader

from data_utils import (Dataset_ASVspoof2019_train,
                        Dataset_ASVspoof2019_devNeval, genSpoof_list)

from models.SSLAASIST import SSLAASIST

warnings.filterwarnings("ignore", category=FutureWarning)


def main():

    # loading configuration file
    # config_path='config/AASIST.conf'
    config_path='config/SSLAASIST.conf'

    with open(config_path, "r") as f_json:
        config = json.loads(f_json.read())

    emb_config=config["embd_config"]

    # data loaders for all
    database_path = Path(config["database_path"])
    trn_loader, dev_loader, eval_loader = get_loader(database_path, config)

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))

    # define model architecture
    # model_config = config["model_config"]
    # model = get_model(model_config, device)
    
    # Load model SSL-AASIST
    model = SSLAASIST(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model = model.to(device)
    print('nb_params:', nb_params)
    
    model.load_state_dict(torch.load(config["model_path"], map_location=device))
    print("Model loaded : {}".format(config["model_path"]))

    print("Start embedding extraction...")
    save_embeddings(trn_loader, dev_loader, eval_loader, model,device, Path(emb_config["exp_dir"]))
    print("Done")


# defining data loaders
def get_loader(
        database_path: str,
        config: dict) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement / evaluation"""

    trn_database_path = database_path / "ASVspoof2019_attr17_train"
    dev_database_path = database_path / "ASVspoof2019_attr17_dev"
    eval_database_path = database_path / "ASVspoof2019_attr17_eval"

    trn_list_path = "/scratch/project_2006687/hoangph/DATA/asvspoof2019/asvspoof2019_attr17/ASVspoof2019_attr17_cm_protocols/Train_ASVspoof19_attr17.txt"

    dev_trial_path = "/scratch/project_2006687/hoangph/DATA/asvspoof2019/asvspoof2019_attr17/ASVspoof2019_attr17_cm_protocols/Dev_ASVspoof19_attr17.txt"

    eval_trial_path = "/scratch/project_2006687/hoangph/DATA/asvspoof2019/asvspoof2019_attr17/ASVspoof2019_attr17_cm_protocols/Eval_ASVspoof19_attr17.txt"

    d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path,
                                            is_train=True,
                                            is_eval=False)
    print("no. training files:", len(file_train))

    # the modification is to use train set to obtain the embeddings
    train_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_train,
                                           base_dir=trn_database_path)
    trn_loader = DataLoader(train_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True,
                            collate_fn=None,
                            num_workers=8)

    _, file_dev = genSpoof_list(dir_meta=dev_trial_path,
                                is_train=False,
                                is_eval=False)
    print("no. validation files:", len(file_dev))

    dev_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_dev,
                                            base_dir=dev_database_path)
    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True,
                            collate_fn=None,
                            num_workers=8)

    file_eval = genSpoof_list(dir_meta=eval_trial_path,
                              is_train=False,
                              is_eval=True)
    eval_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_eval,
                                             base_dir=eval_database_path)
    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True,
                             collate_fn=None,
                             num_workers=8)

    return trn_loader, dev_loader, eval_loader


# loading the model
def get_model(model_config: Dict, device: torch.device):
    """define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model


# extract embeddings
def generate_embeddings(
        data_loader: DataLoader,
        model,
        device: torch.device):
    
    model.eval()
    embs = torch.tensor([])
    for batch_x, utt_id in tqdm(data_loader, desc="Generating Embeddings", ncols=100):
        batch_x=batch_x.to(device)
        with torch.no_grad():
            emb, _ = model(batch_x)
            emb = emb.detach().data.cpu()
        embs = torch.cat((embs,emb))
    return embs.numpy()


# generating embeddings and saving in a folder
def save_embeddings(
        trn_loader, dev_loader, eval_loader,
        model,
        device: torch.device,
        emb_path):
    
    os.makedirs(emb_path, exist_ok=True)
    
    print("Start embedding extraction for train set...")
    embs_train=generate_embeddings(trn_loader,model,device)
    print("Train embeddings generated.")

    print("Start embedding extraction for dev set...")
    embs_dev=generate_embeddings(dev_loader,model,device)
    print("Dev embeddings generated.")

    print("Start embedding extraction for eval set...")
    embs_eval=generate_embeddings(eval_loader,model,device)
    print("Eval embeddings generated.")

    # save in files
    with open(emb_path/"asvspoof2019_attr17_ST_RB_sslaasist_train_emb.npy",'wb') as f:
        np.save(f, embs_train)

    with open(emb_path/"asvspoof2019_attr17_ST_RB_sslaasist_dev_emb.npy",'wb') as f:
        np.save(f, embs_dev)

    with open(emb_path/"asvspoof2019_attr17_ST_RB_sslaasist_eval_emb.npy",'wb') as f:
        np.save(f, embs_eval)


if __name__ == '__main__':
    main()