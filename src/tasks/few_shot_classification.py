import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100, Omniglot


@dataclass
class FewShotClassificationConfig:
    root: str = "./data"
    ways: int = 5
    shots: int = 1
    queries: int = 15
    split: str = "train"  # "train" or "test"
    dataset: str = "cifarfs"


class FewShotClassificationTask:
    """
    Samples few-shot classification episodes from configurable vision datasets.
    Supports CIFAR-FS and Omniglot datasets.
    """

    def __init__(self, config: FewShotClassificationConfig) -> None:
        self.config = config  # Store the configuration
        dataset_name = config.dataset.lower()  # Normalize dataset name
        self.input_resolution: int  # Image height/width
        if dataset_name == "cifarfs":
            train = config.split == "train"
            transform = T.Compose(
                [
                    T.ToTensor(),  # Convert PIL image to tensor
                    T.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),  # CIFAR-FS normalization
                ]
            )
            self.dataset: Dataset = CIFAR100(
                root=config.root,  # Data storage root
                train=train,  # Use training or test split
                download=True,  # Download if not present
                transform=transform,  # Apply transformations
            )
            self.input_channels = 3.  # RGB images
            self.input_resolution = 32  # CIFAR image size
        elif dataset_name == "omniglot":
            background = config.split == "train"
            transform = T.Compose(
                [
                    T.Resize(84),  # Resize images to 84x84
                    T.ToTensor(),  # Convert PIL image to tensor
                    T.Normalize(mean=[0.5], std=[0.5]),  # Omniglot normalization
                ]
            )
            self.dataset = Omniglot(
                root=config.root,
                background=background,  # Use background (train) or evaluation (test) set
                download=True,
                transform=transform,
            )
            self.input_channels = 1
            self.input_resolution = 84
        else:
            raise ValueError(f"Unsupported dataset '{config.dataset}'.")

        self.class_to_indices: Dict[int, List[int]] = {}  # Map from class label to list of dataset indices
        for idx, (_, target) in enumerate(self.dataset):
            self.class_to_indices.setdefault(target, []).append(idx)  # Group indices by class label
        self.available_classes = list(self.class_to_indices.keys())  # List of all class labels

    def sample_episode(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        cfg = self.config
        selected_classes = random.sample(self.available_classes, cfg.ways)
        support_images: List[torch.Tensor] = []  # List of support images
        support_labels: List[int] = []  # List of support labels
        query_images: List[torch.Tensor] = []  # List of query images
        query_labels: List[int] = []  # List of query labels
        local_classes: List[int] = []  # List of original class IDs corresponding to local labels
        """Samples a single few-shot classification episode.
        Returns:
            support_tensor: (ways * shots, C, H, W) tensor of support images
            support_targets: (ways * shots,) tensor of support labels (0 to ways-1)
            query_tensor: (ways * queries, C, H, W) tensor of query images
            query_targets: (ways * queries,) tensor of query labels (0 to ways-1)
            local_classes: List of original class IDs corresponding to each local label
        """

        for local_id, class_id in enumerate(selected_classes):  # Map to local labels 0..ways-1
            indices = random.sample(self.class_to_indices[class_id], cfg.shots + cfg.queries)  # Sample without replacement from class
            for idx in indices[: cfg.shots]:  # iterate over first class indices for support set
                image, _ = self.dataset[idx]  # Retrieve image (ignore original label)
                support_images.append(image)  # Append to support set
                support_labels.append(local_id)  # Append local label
            for idx in indices[cfg.shots :]:  # Remaining indices for query set
                image, _ = self.dataset[idx]  # Retrieve image (ignore original label)
                query_images.append(image)  # Append to query set
                query_labels.append(local_id)  # Append local label
            local_classes.append(class_id)  # Map local label to original class ID

        support_tensor = torch.stack(support_images)  # (ways * shots, C, H, W), C = input channels, H = W = input resolution
        query_tensor = torch.stack(query_images)  # (ways * queries, C, H, W)
        support_targets = torch.tensor(support_labels, dtype=torch.long) # (ways * shots,)
        query_targets = torch.tensor(query_labels, dtype=torch.long) # (ways * queries,)
        return support_tensor, support_targets, query_tensor, query_targets, local_classes  # return the episode data
