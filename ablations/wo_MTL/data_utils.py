import torch
from torch.utils.data import Dataset


class AttClsDataset(Dataset):
    def __init__(self, embeddings, attack_labels):
        """
        Initialize the dataset with embeddings and ground truth labels.
        Args:
            embeddings (np.ndarray): Array of embeddings.
            attack_labels  (np.ndarray): Array of attack labels.
        """
        self.embeddings = embeddings
        self.attack_labels = attack_labels
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, index):
        """
        Get the item at the specified index.
        Args:
            idx (int): Index of the item to retrieve.
        Returns:
            tuple: (embedding, attack_label) for the specified index.
        """
        embedding = torch.tensor(self.embeddings[index], dtype=torch.float32)
        attack_label = torch.tensor(self.attack_labels[index], dtype=torch.long)
        
        return embedding, attack_label
    

ASVSpoof2019_Attr17_attack_attribute_structure = [
    ([0], [0, 1, 2, 3, 6, 7, 8, 9, 10, 11]),
    ([1], [4, 5, 15, 16]),
    ([2], [12, 13, 14]),
    ([3], [0, 1, 2, 3, 6, 7, 8, 11]),
    ([4], [4, 12, 15]),
    ([5], [5]),
    ([6], [9, 10]),
    ([7], [13, 14]),
    ([8], [16]),
    ([9], [0, 1, 7]),
    ([10], [2]),
    ([11], [6, 8, 11]),
    ([12], [9, 10]),
    ([13], [12]),
    ([14], [3, 4, 5, 13, 14, 15, 16]),
    ([15], [0, 1, 7]),
    ([16], [2]),
    ([17], [3]),
    ([18], [4, 15]),
    ([19], [5]),
    ([20], [6, 8, 11, 13, 14]),
    ([21], [9, 10]),
    ([22], [12]),
    ([23], [16]),
    ([24], [0, 1]),
    ([25], [2, 4, 6, 7, 8, 11, 15]),
    ([26], [9, 10]),
    ([27], [16]),
    ([28], [3, 5, 12, 13, 14]),
    ([29], [0, 7, 8, 14, 15]),
    ([30], [1, 2, 13]),
    ([31], [3]),
    ([32], [4]),
    ([33], [5]),
    ([34], [6]),
    ([35], [9, 10]),
    ([36], [11]),
    ([37], [12]),
    ([38], [16]),
    ([39], [0, 11, 14]),
    ([40], [1, 2, 4, 6]),
    ([41], [3]),
    ([42], [5]),
    ([43], [7]),
    ([44], [8]),
    ([45], [9]),
    ([46], [10]),
    ([47], [12, 15]),
    ([48], [13]),
    ([49], [16])
]