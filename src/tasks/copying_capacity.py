from dataclasses import dataclass
from typing import Iterator, List, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class CapacityTaskConfig:
    seq_low_length: int = 5 # Length of the random sequence to be copied
    seq_high_length: int = 10
    delay: int = 5 # Number of delay steps before recall
    vocab_size: int = 8 # Size of the vocabulary
    dataset_size: int = 2000 # Number of samples in the dataset, for generating data
    device: str = "cpu" # Device to run the model on

