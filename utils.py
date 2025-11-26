import os
import random
import sys

import numpy as np
import torch


def seed_worker(worker_id):
    """
    Used in generating seed for the worker of torch.utils.data.Dataloader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(config=None):
    """ 
    set initial seed for reproduction
    """
    if config is None:
        raise ValueError("config should not be None")

    seed = config['seed']
    
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = config["cudnn_deterministic_toggle"]
        torch.backends.cudnn.benchmark = config["cudnn_benchmark_toggle"]
