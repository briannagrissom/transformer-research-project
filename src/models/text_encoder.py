"""
Character-level text encoder for converting strings to fixed-size embeddings.

This encoder is one of several options for the few-shot text classification pipeline.
It represents words based on their character composition, meaning:
  - Morphologically similar words (e.g., "cat" vs "car") get similar embeddings
  - The encoder can handle any string, even words not in its training vocabulary
  - It does NOT inherently capture semantic meaning ("dog" and "cat" may be distant
    despite being semantically related animals)

The encoder uses a bidirectional LSTM to process character sequences, producing
a fixed-size vector regardless of input string length. This vector is then fed
into the plastic transformer as part of the few-shot learning pipeline.
"""
import torch
import torch.nn as nn
import string


class CharacterEncoder(nn.Module):
    """
    Encodes strings as character-level embeddings using a bidirectional LSTM.
    
    Processing pipeline:
      1. String -> character indices (lookup table for a-z, space, special tokens)
      2. Character indices -> character embeddings (learned embedding table)
      3. Character embeddings -> LSTM hidden states (bidirectional, captures context)
      4. Final hidden states (forward + backward) -> linear projection -> output embedding
    
    In the few-shot classification context:
      - This encoder is trained end-to-end with the plastic transformer
      - Gradients flow from the classification loss all the way back through the LSTM
        to the character embeddings, so the encoder learns to produce representations
        that are useful for the downstream few-shot task
      - Since it's character-based, it can encode ANY string (not limited to a vocabulary)
    """
    
    def __init__(
        self,
        output_dim: int = 256,
        char_embed_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            output_dim: Dimensionality of the output embedding vector
            char_embed_dim: Dimensionality of character embeddings
            hidden_dim: Hidden size of the LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()
        
        # Build character vocabulary: lowercase letters + space + special tokens
        # Special tokens: [PAD], [UNK] (unknown character)
        self.chars = ['[PAD]', '[UNK]'] + list(string.ascii_lowercase) + [' ']
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.pad_idx = 0
        self.unk_idx = 1
        
        # Character embedding layer
        self.char_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=char_embed_dim,
            padding_idx=self.pad_idx
        )
        
        # LSTM to process character sequences
        self.lstm = nn.LSTM(
            input_size=char_embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True  # Use bidirectional LSTM for better context
        )
        
        # Project LSTM output to desired embedding dimension
        # *2 because bidirectional LSTM outputs 2*hidden_dim
        self.projection = nn.Linear(hidden_dim * 2, output_dim)
        
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
    def encode_strings(self, strings: list[str], max_length: int = 50) -> torch.Tensor:
        """
        Convert list of strings to tensor of character indices.
        
        Args:
            strings: List of strings to encode
            max_length: Maximum sequence length (will pad/truncate)
            
        Returns:
            Tensor of shape (batch_size, max_length) with character indices
        """
        batch_size = len(strings)
        indices = torch.full((batch_size, max_length), self.pad_idx, dtype=torch.long)
        
        for i, s in enumerate(strings):
            # Convert to lowercase and encode each character
            s_lower = s.lower()
            for j, ch in enumerate(s_lower[:max_length]):
                indices[i, j] = self.char_to_idx.get(ch, self.unk_idx)
        
        return indices
    
    def forward(self, strings: list[str]) -> torch.Tensor:
        """
        Encode strings into fixed-size embeddings.
        
        Args:
            strings: List of strings to encode
            
        Returns:
            Tensor of shape (batch_size, output_dim) with string embeddings
        """
        # Convert strings to character indices
        char_indices = self.encode_strings(strings)  # (batch_size, max_length)
        char_indices = char_indices.to(self.char_embedding.weight.device)
        
        # Embed characters
        char_embeds = self.char_embedding(char_indices)  # (batch_size, max_length, char_embed_dim)
        
        # Process with LSTM
        lstm_out, (hidden, _) = self.lstm(char_embeds)
        # lstm_out: (batch_size, max_length, hidden_dim * 2)
        # hidden: (num_layers * 2, batch_size, hidden_dim) for bidirectional
        
        # Use the final hidden state from both directions
        # Concatenate forward and backward final hidden states from the last layer
        forward_hidden = hidden[-2]  # (batch_size, hidden_dim)
        backward_hidden = hidden[-1]  # (batch_size, hidden_dim)
        final_hidden = torch.cat([forward_hidden, backward_hidden], dim=-1)  # (batch_size, hidden_dim * 2)
        
        # Project to output dimension
        embeddings = self.projection(final_hidden)  # (batch_size, output_dim)
        
        return embeddings
    
    def forward_batch(self, strings: list[str]) -> torch.Tensor:
        """
        Convenience method that processes strings in batch mode.
        Same as forward() but with explicit batching semantics.
        
        Args:
            strings: List of strings to encode
            
        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        return self.forward(strings)
