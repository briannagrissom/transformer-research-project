"""
Baseline model using pre-trained transformers for comparison.

This provides a frozen pre-trained transformer baseline to compare against
the plastic transformer's in-context learning capabilities.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class PretrainedTransformerBaseline(nn.Module):
    """
    Uses a frozen pre-trained transformer as a feature extractor.
    
    For few-shot learning, this processes all support + query examples
    through the transformer, then uses a simple classifier head.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        output_dim: int = 256,
        freeze_backbone: bool = True,
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            output_dim: Dimension for the classification head
            freeze_backbone: Whether to freeze the pre-trained weights
        """
        super().__init__()
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, use_safetensors=True)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get the hidden size from the backbone
        self.hidden_size = self.backbone.config.hidden_size
        
        # Projection head
        self.projection = nn.Linear(self.hidden_size, output_dim)
        
    def encode_strings(self, strings: list[str], device: torch.device) -> torch.Tensor:
        """
        Encode strings using the pre-trained transformer.
        
        Args:
            strings: List of strings to encode
            device: Device to run on
            
        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        # Tokenize
        encoded = self.tokenizer(
            strings,
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors="pt"
        ).to(device)
        
        # Get [CLS] token embeddings
        with torch.no_grad():
            outputs = self.backbone(**encoded)
            # Use [CLS] token (first token) as sentence representation
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Project to output dimension
        embeddings = self.projection(cls_embeddings)
        
        return embeddings
    
    def forward_episode(
        self,
        support_strings: list[str],
        support_labels: torch.Tensor,
        query_strings: list[str],
        device: torch.device,
        ways: int,
    ) -> torch.Tensor:
        """
        Process an episode using prototype-based classification.
        
        Args:
            support_strings: Support set strings
            support_labels: Support set labels (class indices)
            query_strings: Query set strings
            device: Device to run on
            ways: Number of classes
            
        Returns:
            Logits of shape (num_queries, ways)
        """
        # Encode support and query
        support_embeddings = self.encode_strings(support_strings, device)
        query_embeddings = self.encode_strings(query_strings, device)
        
        # Compute class prototypes (mean of support examples per class)
        prototypes = []
        for class_idx in range(ways):
            mask = support_labels == class_idx
            if mask.sum() > 0:
                class_embeddings = support_embeddings[mask]
                prototype = class_embeddings.mean(dim=0)
                prototypes.append(prototype)
            else:
                # No examples for this class - use zero vector
                prototypes.append(torch.zeros(support_embeddings.shape[1], device=device))
        
        prototypes = torch.stack(prototypes)  # Shape: (ways, output_dim)
        
        # Compute distances to prototypes (negative L2 distance = similarity)
        # query_embeddings: (num_queries, output_dim)
        # prototypes: (ways, output_dim)
        distances = torch.cdist(query_embeddings, prototypes, p=2)  # (num_queries, ways)
        
        # Convert distances to logits (negative distance = higher similarity)
        logits = -distances
        
        return logits


class PretrainedBaselineForWordCategory(nn.Module):
    """
    Wrapper for using pre-trained transformer on word category classification.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        output_dim: int = 256,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.baseline = PretrainedTransformerBaseline(
            model_name=model_name,
            output_dim=output_dim,
            freeze_backbone=freeze_backbone,
        )
    
    def forward(
        self,
        support_strings: list[str],
        support_labels: list[int],
        query_strings: list[str],
        ways: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Process an episode.
        
        Returns:
            Logits of shape (num_queries, ways)
        """
        support_labels_tensor = torch.tensor(support_labels, dtype=torch.long, device=device)
        return self.baseline.forward_episode(
            support_strings=support_strings,
            support_labels=support_labels_tensor,
            query_strings=query_strings,
            device=device,
            ways=ways,
        )
