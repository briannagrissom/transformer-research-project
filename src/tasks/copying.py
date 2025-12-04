from dataclasses import dataclass
from typing import Iterator, List, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class CopyingTaskConfig:
    seq_length: int = 5 # Length of the random sequence to be copied
    delay: int = 5 # Number of delay steps before recall
    vocab_size: int = 8 # Size of the vocabulary
    dataset_size: int = 2000 # Number of samples in the dataset, for generating data
    device: str = "cpu" # Device to run the model on


class CopyingTaskDataset(Dataset):
    """
    Generates sequences for the copying task. Each trial consists of:

    - seq_length random tokens sampled from {1, ..., vocab_size - 2}
    - delay tokens of the blank symbol (0)
    - a delimiter token (vocab_size - 1) signalling recall
    - seq_length blank tokens where the target is the original sequence
    """

    def __init__(self, config: CopyingTaskConfig) -> None:
        self.config = config # Store the configuration
        self.total_length = config.seq_length * 2 + config.delay + 1 # Total length of each sequence, which includes the input sequence, delay, delimiter, and recall phase

    def __len__(self) -> int:
        return self.config.dataset_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a single copying task trial.
        Returns (inputs, targets) where both are sequences of token indices.
        """
        cfg = self.config
        
        # PHASE 1: Generate random sequence to memorize
        # Sample tokens from {1, 2, ..., vocab_size-2}
        # Token 0 is reserved for blank, token vocab_size-1 is reserved for delimiter
        seq = torch.randint(
            low=1,
            high=cfg.vocab_size - 1,
            size=(cfg.seq_length,),
            dtype=torch.long,
        )
        
        # PHASE 2: Create delay period filled with blank tokens (0)
        # This tests if the model can maintain memory over time
        blank = torch.zeros(cfg.delay, dtype=torch.long)
        
        # PHASE 3: Create delimiter token to signal "start recalling now!"
        # Uses the highest token value: vocab_size - 1
        delimiter = torch.full((1,), cfg.vocab_size - 1, dtype=torch.long)
        
        # PHASE 4: Create recall phase inputs (all blanks)
        # Model must output the original sequence despite blank inputs
        recall_inputs = torch.zeros(cfg.seq_length, dtype=torch.long)

        # Concatenate all four phases to create the full input sequence
        # Result: [seq | blank | delimiter | recall_inputs]
        inputs = torch.cat([seq, blank, delimiter, recall_inputs], dim=0)

        # Create target sequence:
        # - During presentation, delay, and delimiter: target is blank (0)
        # - During recall phase: target is the original sequence
        # This means we only evaluate the model's predictions during the recall phase
        blank_targets = torch.zeros(cfg.seq_length + cfg.delay + 1, dtype=torch.long)
        targets = torch.cat([blank_targets, seq], dim=0)
        return inputs, targets


def generate_copying_batch(
    dataset: CopyingTaskDataset,
    batch_indices: List[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a batch of copying task sequences.
    Collects multiple trials and stacks them into batch tensors.
    
    Args:
        dataset: The copying task dataset to sample from
        batch_indices: List of indices to retrieve from the dataset
        device: Device to move the tensors to (cpu/cuda/mps)
    
    Returns:
        Tuple of (inputs_tensor, targets_tensor) with batch dimension added
    """
    inputs_batch = []
    targets_batch = []
    
    # Collect all sequences in the batch
    for idx in batch_indices:
        inputs, targets = dataset[idx]
        inputs_batch.append(inputs)
        targets_batch.append(targets)
    
    # Stack individual sequences into batch tensors and move to device
    # Shape: (batch_size, sequence_length)
    inputs_tensor = torch.stack(inputs_batch).to(device)
    targets_tensor = torch.stack(targets_batch).to(device)
    return inputs_tensor, targets_tensor

