
# Vanilla FedAvg Implementation
# Reference: McMahan et al., "Communication-Efficient Learning of Deep Networks
# from Decentralized Data", AISTATS 2017
from src.algorithms.base import GenericAlgorithm

class VanillaFedAvg(GenericAlgorithm):
    """Standard Federated Averaging — no privacy, no robustness enhancements."""
    pass
