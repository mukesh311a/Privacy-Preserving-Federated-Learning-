
# Dataset Generator for HE-FL-IoT Framework
# Simulates Non-IID healthcare data partitioning.

import numpy as np
from typing import Dict, List

class DatasetGenerator:
    """Simulates Non-IID Healthcare IoT Dataset Partitioning using Dirichlet distribution."""
    
    def __init__(self, num_clients: int = 20, alpha: float = 0.5, seed: int = 42):
        self.num_clients = num_clients
        self.alpha = alpha  # Dirichlet concentration parameter
        self.seed = seed
        np.random.seed(seed)
    
    def generate_partition_stats(self, num_classes: int = 10, total_samples: int = 50000) -> Dict:
        """Generates per-client data distribution statistics."""
        
        # Dirichlet distribution for label distribution across clients
        label_distribution = np.random.dirichlet(
            [self.alpha] * num_classes, self.num_clients
        )
        
        # Assign samples to clients
        samples_per_client = np.random.multinomial(
            total_samples, np.ones(self.num_clients) / self.num_clients
        )
        
        client_stats = {}
        for c in range(self.num_clients):
            client_stats[f"Client_{c+1}"] = {
                "num_samples": int(samples_per_client[c]),
                "label_distribution": label_distribution[c].tolist(),
                "dominant_class": int(np.argmax(label_distribution[c])),
                "entropy": float(-np.sum(label_distribution[c] * np.log(label_distribution[c] + 1e-10)))
            }
        
        return {
            "num_clients": self.num_clients,
            "alpha": self.alpha,
            "num_classes": num_classes,
            "total_samples": total_samples,
            "client_stats": client_stats
        }
