"""
Word2Vec-based text encoder using pre-trained embeddings.
Uses GloVe embeddings for semantic word representations.

This encoder is one of several options for the few-shot text classification pipeline.
Unlike the character-level encoder, this one uses pre-trained GloVe word vectors
that already capture rich semantic relationships:
  - Semantically similar words have similar vectors (e.g., "dog" and "cat" are close)
  - Different categories tend to form clusters in the embedding space

This gives the plastic transformer a semantic "head start" -- words in the same
category (e.g., animals) already have similar representations before any plastic
adaptation happens. The transformer's job is then to learn a decision boundary
between the categories presented in each episode.
"""
import os
import numpy as np
import torch
import torch.nn as nn


class Word2VecEncoder(nn.Module):
    """
    Encodes strings using pre-trained Word2Vec/GloVe embeddings.
    
    Processing pipeline:
      1. Look up word in the pre-trained GloVe vocabulary
      2. Retrieve the fixed GloVe vector (e.g., 100-dimensional)
      3. Project through a learned linear layer to the desired output_dim
    
    The GloVe vectors are FROZEN (not updated during training), but the projection
    layer IS learned. This means the model can learn to emphasize dimensions of the
    GloVe space that are most useful for few-shot category classification.
    
    For the few-shot task, this encoder's semantic structure means that after seeing
    just one example of "dog" labeled as category 0, the model can potentially
    recognize "cat" as the same category because their GloVe embeddings are similar.
    """
    
    def __init__(
        self,
        vocabulary: list[str],
        output_dim: int = 256,
        glove_path: str = None,
        glove_dim: int = 50,
    ):
        """
        Args:
            vocabulary: List of words in the vocabulary
            output_dim: Dimensionality of the output embedding vector
            glove_path: Path to GloVe embeddings file (e.g., glove.6B.50d.txt)
            glove_dim: Dimension of GloVe embeddings (50, 100, 200, or 300)
        """
        super().__init__()
        
        self.vocabulary = vocabulary
        self.output_dim = output_dim
        self.glove_dim = glove_dim
        
        # Load GloVe embeddings
        print(f"Loading GloVe embeddings from {glove_path}...")
        self.glove_embeddings = self._load_glove_embeddings(glove_path)
        print(f"Loaded {len(self.glove_embeddings)} GloVe vectors")
        
        # Create embedding matrix for vocabulary
        self.word_embeddings = self._create_embedding_matrix()
        
        # Learnable projection from GloVe dimension to output dimension
        self.projection = nn.Linear(glove_dim, output_dim)
        
        # Learnable embedding for unknown words
        self.unk_embedding = nn.Parameter(torch.randn(glove_dim))
    
    def _load_glove_embeddings(self, glove_path: str) -> dict:
        """Load GloVe embeddings from file."""
        embeddings = {}
        
        if glove_path is None or not os.path.exists(glove_path):
            print(f"Warning: GloVe file not found at {glove_path}")
            print("Using random embeddings as fallback")
            return embeddings
        
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings[word] = vector
        
        return embeddings
    
    def _create_embedding_matrix(self) -> nn.Parameter:
        """Create embedding matrix for vocabulary words."""
        embedding_matrix = torch.zeros(len(self.vocabulary), self.glove_dim)
        
        found_count = 0
        for idx, word in enumerate(self.vocabulary):
            word_lower = word.lower()
            if word_lower in self.glove_embeddings:
                embedding_matrix[idx] = torch.from_numpy(self.glove_embeddings[word_lower])
                found_count += 1
            else:
                # Initialize with random vector for unknown words
                embedding_matrix[idx] = torch.randn(self.glove_dim) * 0.1
        
        print(f"Found {found_count}/{len(self.vocabulary)} words in GloVe vocabulary")
        
        # Make it a parameter so it moves to the right device
        return nn.Parameter(embedding_matrix, requires_grad=False)
    
    def forward(self, strings: list[str]) -> torch.Tensor:
        """
        Encode strings into fixed-size embeddings.
        
        Args:
            strings: List of strings to encode
            
        Returns:
            Tensor of shape (batch_size, output_dim) with string embeddings
        """
        batch_size = len(strings)
        embeddings = torch.zeros(batch_size, self.glove_dim, device=self.projection.weight.device)
        
        for i, string in enumerate(strings):
            # Convert to lowercase
            word = string.lower()
            
            # Look up in vocabulary
            if word in self.vocabulary:
                idx = self.vocabulary.index(word)
                embeddings[i] = self.word_embeddings[idx]
            else:
                # Use unknown embedding
                embeddings[i] = self.unk_embedding
        
        # Project to output dimension
        output = self.projection(embeddings)
        
        return output


class SimpleSemanticEncoder(nn.Module):
    """
    Simple semantic encoder that manually creates embeddings with semantic structure.
    
    This is a lightweight alternative to GloVe when pre-trained embeddings are not
    available. It creates embeddings where:
      - Words in the same category share a common base vector (learned)
      - Each word has a small individual offset (learned) to distinguish it within
        its category
    
    The category base vectors are initialized with high magnitude and normalized, so
    different categories are well-separated in the embedding space from the start.
    
    For the few-shot task, this encoder explicitly encodes category membership into
    the representation, making the plastic transformer's job easier -- it just needs
    to learn to read the category signal from the embedding.
    """
    
    def __init__(
        self,
        vocabulary: list[str],
        categories: dict[str, list[str]],
        output_dim: int = 256,
    ):
        """
        Args:
            vocabulary: List of all words
            categories: Dictionary mapping category names to word lists
            output_dim: Dimensionality of the output embedding vector
        """
        super().__init__()
        
        self.vocabulary = vocabulary
        self.categories = categories
        self.output_dim = output_dim
        
        # Create word to category mapping
        self.word_to_category = {}
        for category, words in categories.items():
            for word in words:
                self.word_to_category[word] = category
        
        # Create learnable embeddings
        # Each category gets a base vector, each word adds a small perturbation
        self.category_base = nn.Parameter(torch.randn(len(categories), output_dim))
        self.word_offset = nn.Parameter(torch.randn(len(vocabulary), output_dim) * 0.1)
        
        # Normalize to make categories well-separated
        with torch.no_grad():
            self.category_base.data = torch.nn.functional.normalize(
                self.category_base.data, p=2, dim=1
            ) * 5.0  # Scale up to increase separation
    
    def forward(self, strings: list[str]) -> torch.Tensor:
        """
        Encode strings into fixed-size embeddings.
        
        Args:
            strings: List of strings to encode
            
        Returns:
            Tensor of shape (batch_size, output_dim) with string embeddings
        """
        batch_size = len(strings)
        embeddings = torch.zeros(batch_size, self.output_dim, device=self.category_base.device)
        
        category_to_idx = {cat: idx for idx, cat in enumerate(self.categories.keys())}
        
        for i, string in enumerate(strings):
            word = string.lower()
            
            if word in self.vocabulary:
                # Get category base + word-specific offset
                category = self.word_to_category.get(word)
                if category:
                    cat_idx = category_to_idx[category]
                    word_idx = self.vocabulary.index(word)
                    embeddings[i] = self.category_base[cat_idx] + self.word_offset[word_idx]
                else:
                    # Unknown category, just use word offset
                    word_idx = self.vocabulary.index(word)
                    embeddings[i] = self.word_offset[word_idx]
            else:
                # Unknown word - use random
                embeddings[i] = torch.randn(self.output_dim, device=embeddings.device) * 0.1
        
        return embeddings
