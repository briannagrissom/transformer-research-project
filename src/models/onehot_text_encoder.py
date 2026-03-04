"""
Simple one-hot text encoder for debugging.
Maps each word to a unique one-hot vector.

This encoder is one of several options for the few-shot text classification pipeline.
It provides NO built-in semantic structure -- each word gets a randomly initialized,
independent embedding vector. This makes it a useful baseline and debugging tool:
  - If the model succeeds with this encoder, it means the plastic transformer is
    genuinely learning to map arbitrary embeddings to categories from few examples
  - If the model fails with this encoder but succeeds with semantic encoders,
    it suggests the model relies on pre-existing similarity structure in the embeddings

The embedding vectors ARE learned during training (via outer-loop optimization),
so the model can learn to place same-category words near each other over time.
However, at the start of training, there's no inherent similarity between words
in the same category.
"""
import torch
import torch.nn as nn


class OneHotTextEncoder(nn.Module):
    """
    Encodes strings as learned embeddings based on a fixed vocabulary.
    
    Processing pipeline:
      1. Map each input word to its vocabulary index (or unknown token)
      2. Look up the corresponding embedding vector from a learned embedding table
    
    Despite the name "OneHot", this doesn't output literal one-hot vectors.
    It uses an nn.Embedding layer which maps discrete indices to dense vectors.
    The name reflects that each word is treated as an independent identity
    (no character-level or semantic similarity built in).
    
    In the few-shot context, this encoder starts with random embeddings, so
    the plastic transformer cannot exploit any pre-existing word similarity.
    Over outer-loop training, the embeddings may learn to cluster same-category
    words, but this is entirely learned rather than pre-specified.
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
