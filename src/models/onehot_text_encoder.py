"""
Simple one-hot text encoder for debugging.
Maps each word to a unique one-hot vector.
"""
import torch
import torch.nn as nn


class OneHotTextEncoder(nn.Module):
    """
    Encodes strings as one-hot vectors based on a fixed vocabulary.
    
    This is a simple encoder for debugging that guarantees each unique word
    gets a completely different embedding.
    """
    
    def __init__(
        self,
        vocabulary: list[str],
        output_dim: int = 256,
    ):
        """
        Args:
            vocabulary: List of words in the vocabulary
            output_dim: Dimensionality of the output embedding vector
        """
        super().__init__()
        
        self.vocabulary = vocabulary
        self.word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
        self.vocab_size = len(vocabulary)
        self.output_dim = output_dim
        
        # Learnable embedding matrix: maps word index to embedding
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size + 1,  # +1 for unknown words
            embedding_dim=output_dim
        )
        
        self.unk_idx = self.vocab_size  # Unknown word index
    
    def forward(self, strings: list[str]) -> torch.Tensor:
        """
        Encode strings into fixed-size embeddings.
        
        Args:
            strings: List of strings to encode
            
        Returns:
            Tensor of shape (batch_size, output_dim) with string embeddings
        """
        batch_size = len(strings)
        device = self.embedding.weight.device
        
        # Convert strings to indices
        indices = []
        for s in strings:
            s_lower = s.lower().strip()
            idx = self.word_to_idx.get(s_lower, self.unk_idx)
            indices.append(idx)
        
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=device)
        
        # Look up embeddings
        embeddings = self.embedding(indices_tensor)  # (batch_size, output_dim)
        
        return embeddings
