
# HE-FL-IoT Simulation Framework
# Privacy-Preserving Federated Learning with CKKS Homomorphic Encryption
# for Secure Healthcare IoT Analytics

import os
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class SimulationConfig:
    """Configuration for the HE-FL-IoT Simulation."""
    
    # General settings
    NUM_ROUNDS: int = 200
    NUM_SEEDS: int = 10  # Increased for Reviewer 2
    REFERENCE_BASELINE_ID: int = 42
    
    # Hardware/Environment Metadata (for Reviewer 2)
    HARDWARE_SPEC: str = "High-Performance Compute Node (Intel Core i7-12700H, 32GB RAM)"
    OS_SPEC: str = "Windows 11 / Python 3.10"
    
    # Sensitivity Analysis Space (for Reviewer 1)
    P_SEARCH_SPACE: List[int] = field(default_factory=lambda: [25, 50, 75, 90, 95])
    
    # Output paths
    OUTPUT_DIR: str = "results"
    METRICS_FILE: str = "comprehensive_metrics.json"
    RADAR_FILE: str = "radar_metrics.json"
    STATS_REPORT: str = "statistical_report.txt"
    
    # Visualization settings
    PLOT_STYLE: str = "seaborn-v0_8-whitegrid"
    DPI: int = 300
    
    # Algorithm Registry (8 baselines + 3 ablation + 1 proposed = 12)
    ALGORITHMS: List[str] = field(default_factory=lambda: [
        # --- Baselines ---
        "Vanilla FedAvg",
        "FedProx",
        "SCAFFOLD",
        "DP-FedAvg",
        "FedBN",
        "FedNova",
        "FLASHE",
        "BatchCrypt",
        # --- Ablation Variants ---
        "FL + CKKS HE Only",
        "FL + Gradient Compression Only",
        "FL + Adaptive DP Only",
        # --- Proposed ---
        "HE-FL-IoT (Proposed)"
    ])

    def __post_init__(self):
        if not os.path.exists(self.OUTPUT_DIR):
            os.makedirs(self.OUTPUT_DIR)
