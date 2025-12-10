"""
Test script to check embedding distinguishability for word categories.
"""
import torch
import torch.nn.functional as F
from src.models.text_encoder import CharacterEncoder
from src.models.onehot_text_encoder import OneHotTextEncoder
from src.tasks.word_category_classification import CATEGORIES, ALL_WORDS

def compute_cosine_similarity_matrix(embeddings):
    """Compute pairwise cosine similarities."""
    # Normalize embeddings
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    # Compute cosine similarity matrix
    similarity = torch.mm(embeddings_norm, embeddings_norm.t())
    return similarity

def analyze_embeddings(encoder, encoder_name):
    """Analyze how distinguishable embeddings are within and across categories."""
    print(f"\n{'='*70}")
    print(f"Analyzing {encoder_name}")
    print(f"{'='*70}\n")
    
    # Get all words organized by category
    all_words_ordered = []
    category_indices = {}
    start_idx = 0
    
    for category, words in CATEGORIES.items():
        all_words_ordered.extend(words)
        category_indices[category] = list(range(start_idx, start_idx + len(words)))
        start_idx += len(words)
    
    # Encode all words
    with torch.no_grad():
        embeddings = encoder(all_words_ordered)
    
    # Compute similarity matrix
    similarity = compute_cosine_similarity_matrix(embeddings)
    
    # Analyze within-category similarity
    print("Within-category similarities (should be LOW for good distinguishability):")
    for category, indices in category_indices.items():
        within_sims = []
        for i, idx1 in enumerate(indices):
            for idx2 in indices[i+1:]:
                within_sims.append(similarity[idx1, idx2].item())
        
        if within_sims:
            avg_sim = sum(within_sims) / len(within_sims)
            max_sim = max(within_sims)
            min_sim = min(within_sims)
            print(f"  {category:12s}: avg={avg_sim:.3f}, max={max_sim:.3f}, min={min_sim:.3f}")
    
    # Analyze cross-category similarity
    print("\nCross-category similarities (should be LOW for good separation):")
    category_pairs = [
        ("animals", "clothing"),
        ("animals", "furniture"),
        ("clothing", "furniture"),
    ]
    
    for cat1, cat2 in category_pairs:
        indices1 = category_indices[cat1]
        indices2 = category_indices[cat2]
        
        cross_sims = []
        for idx1 in indices1:
            for idx2 in indices2:
                cross_sims.append(similarity[idx1, idx2].item())
        
        avg_sim = sum(cross_sims) / len(cross_sims)
        max_sim = max(cross_sims)
        print(f"  {cat1:12s} <-> {cat2:12s}: avg={avg_sim:.3f}, max={max_sim:.3f}")
    
    # Show most confusable pairs
    print("\nMost confusable word pairs (highest similarity):")
    n = len(all_words_ordered)
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((similarity[i, j].item(), all_words_ordered[i], all_words_ordered[j]))
    
    pairs.sort(reverse=True)
    for sim, word1, word2 in pairs[:10]:
        # Determine categories
        cat1 = None
        cat2 = None
        for cat, words in CATEGORIES.items():
            if word1 in words:
                cat1 = cat
            if word2 in words:
                cat2 = cat
        
        marker = "⚠️ SAME CATEGORY" if cat1 == cat2 else "✓ diff category"
        print(f"  {sim:.3f}: '{word1:10s}' <-> '{word2:10s}' ({marker})")
    
    # Show embedding norms
    print("\nEmbedding norms:")
    for category, indices in category_indices.items():
        norms = [embeddings[idx].norm().item() for idx in indices]
        avg_norm = sum(norms) / len(norms)
        print(f"  {category:12s}: avg_norm={avg_norm:.3f}")


if __name__ == "__main__":
    # Test CharacterEncoder
    char_encoder = CharacterEncoder(
        output_dim=256,
        char_embed_dim=32,
        hidden_dim=128,
        num_layers=2,
        dropout=0.0,
    )
    char_encoder.eval()
    
    analyze_embeddings(char_encoder, "CharacterEncoder (Bidirectional LSTM)")
    
    # Test OneHotTextEncoder
    onehot_encoder = OneHotTextEncoder(
        vocabulary=ALL_WORDS,
        output_dim=256,
    )
    onehot_encoder.eval()
    
    analyze_embeddings(onehot_encoder, "OneHotTextEncoder (Independent Embeddings)")
    
    # Test SimpleSemanticEncoder
    from src.models.word2vec_encoder import SimpleSemanticEncoder
    
    semantic_encoder = SimpleSemanticEncoder(
        vocabulary=ALL_WORDS,
        categories=CATEGORIES,
        output_dim=256,
    )
    semantic_encoder.eval()
    
    analyze_embeddings(semantic_encoder, "SimpleSemanticEncoder (Category-based Embeddings)")
