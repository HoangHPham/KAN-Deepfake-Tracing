import torch
from torch.utils.data import Dataset


class MultitaskDataset(Dataset):
    def __init__(self, embeddings, ground_truth):
        """
        Initialize the dataset with embeddings and ground truth labels.
        Args:
            embeddings (np.ndarray): Array of embeddings.
            ground_truth  (dict): Dictionary containing ground truth labels for each task.
        """
        self.embeddings = embeddings
        self.ground_truth = ground_truth
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, index):
        """
        Get the item at the specified index.
        Args:
            idx (int): Index of the item to retrieve.
        Returns:
            tuple: (embedding, ground_truth) for the specified index.
        """
        embedding = torch.tensor(self.embeddings[index], dtype=torch.float32)
        attribute_labels = {
            task: torch.tensor(self.ground_truth[index][task], dtype=torch.long).squeeze(0)
                for task in self.ground_truth[index]
        }
        return embedding, attribute_labels