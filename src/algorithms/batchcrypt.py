
# BatchCrypt Implementation
# Reference: Zhang et al., "BatchCrypt: Efficient Homomorphic Encryption for
# Cross-Silo Federated Learning", USENIX ATC 2020
from src.algorithms.base import GenericAlgorithm

class BatchCryptAlgo(GenericAlgorithm):
    """BatchCrypt with batch-level quantization + Paillier encryption."""
    pass
