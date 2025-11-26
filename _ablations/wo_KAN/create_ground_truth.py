"""
Create ground truth for 2 modules:
- Multi-task learning model:
    + Ground-truth includes flag (bonafide or spoof) and attributes for each task (AS1, AS2, etc.)
- KAN model:
    + Ground-truth is the attack type of sample (A01, A02, etc.)
"""

import os
import numpy as np
import pandas as pd


def load_CMembeddings(CMembeddings_path):
    """
    Load 160-dimensional AASIST's embeddings from the specified path.
    Args:
        embeddings_path (str): Path to the embeddings file.
    Returns:
        np.ndarray: Loaded embeddings as a numpy array.
    """
    if not os.path.exists(CMembeddings_path):
        raise FileNotFoundError(f"Embeddings file not found at {CMembeddings_path}")
    embeddings = np.load(CMembeddings_path)
    if embeddings.ndim != 2 or embeddings.shape[1] != 160:
        raise ValueError(f"Embeddings should be a 2D array with 160 dimensions, got shape {embeddings.shape}")
    
    return embeddings.astype(np.float32)  # Ensure the embeddings are in float32 format


def load_protocol_data(protocols_path):
    """
    Load protocol data from the specified path.
    Args:
        protocols_path (str): Path to the protocols file.
    Returns:
        pd.DataFrame: Loaded protocol data as a pandas DataFrame.
    """
    if not os.path.exists(protocols_path):
        raise FileNotFoundError(f"Protocols file not found at {protocols_path}")
    df = pd.read_csv(protocols_path, sep=" ", header=None)
    
    return df


def get_labels(source):

    # data based on "Table 1: Summary of LA spoofing systems." in "ASVspoof 2019: a large-scale public database of synthetized, converted and replayed speech"
    
    # the data is not balanced when it comes to the attributes (e.g. Text(input) has more samples than MCC-F0(output))
    data = {
            "A01":      [[0], [0], [0], [0], [0], [0], [0], [0]],
            "A02":      [[1], [0], [0], [0], [0], [0], [1], [1]],
            "A03":      [[2], [0], [0], [1], [1], [1], [1], [1]],
            "A04-16":   [[3], [0], [0], [5], [2], [4], [2], [2]],
            "A05":      [[4], [1], [1], [5], [3], [1], [3], [1]],
            "A06-19":   [[5], [1], [2], [5], [4], [4], [4], [3]],

            "A07":      [[6], [0], [0], [2], [5], [1], [5], [1]],
            "A08":      [[7], [0], [0], [0], [0], [1], [0], [4]],
            "A09":      [[8], [0], [0], [2], [5], [1], [0], [5]],
            "A10":      [[9], [0], [3], [3], [6], [2], [6], [6]], 
            "A11":      [[10], [0], [3], [3], [6], [2], [6], [7]],
            "A12":      [[11], [0], [0], [2], [5], [1], [7], [0]],
            "A13":      [[12], [2], [1], [4], [7], [4], [8], [8]],
            "A14":      [[13], [2], [4], [5], [5], [4], [1], [9]],
            "A15":      [[14], [2], [4], [5], [5], [4], [0], [0]],
            "A17":      [[15], [1], [1], [5], [3], [1], [0], [8]],
            "A18":      [[16], [1], [5], [5], [8], [3], [9], [10]],
        } 
    
    attribute_sets = ["AS1", "AS2", "AS3", "AS4", "AS5", "AS6", "AS7"]
    
    labels = {}
    for attack_type, ys in data.items():
        labels[attack_type] = {
            "Attack_id": ys[0],
            **{f'T{i+1}_{attribute_sets[i]}': ys[i+1] for i in range(len(attribute_sets))},
        }
    
    return labels[source]


def create_ground_truth(protocols_path):
    """
    Create ground truth for multi-task learning based on embeddings and protocols.
    Args:
        embeddings_path (str): Path to the embeddings file.
        protocols_path (str): Path to the protocols file.
    """
    protocols_data = load_protocol_data(protocols_path)
    
    ground_truth = []
    for index, row in protocols_data.iterrows():
        source = row[3]
        attributes = get_labels(source)
        ground_truth.append(attributes)
    ground_truth = np.array(ground_truth, dtype=object)
    return ground_truth


def load_attack_labels(protocols_path):
    protocols_data = load_protocol_data(protocols_path)
    
    ground_truth = []
    for index, row in protocols_data.iterrows():
        source = row[3]
        attributes = get_labels(source)
        ground_truth.append(attributes['Attack_id'][0])
    ground_truth = np.array(ground_truth, dtype=np.int64)
    return ground_truth