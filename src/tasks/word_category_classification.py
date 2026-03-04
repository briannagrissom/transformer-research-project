"""
Few-shot word category classification task.
Classifies words into semantic categories (animals, clothing, furniture, etc.)

=== OVERVIEW OF FEW-SHOT WORD CATEGORY CLASSIFICATION ===

This module implements a few-shot learning task where the model must learn to
classify words into semantic categories (e.g., "dog" -> animals, "chair" -> furniture)
from only a handful of labeled examples (the "support set"), and then generalize
to classify unseen words from the same categories (the "query set").

Key concepts:
  - "N-way K-shot" episodic learning: Each training/test episode samples N categories
    ("ways") and provides K labeled examples per category ("shots").
  - Support set: The few labeled examples the model sees to learn the task.
  - Query set: The unlabeled examples the model must classify.
  - Episode: A single few-shot learning trial with its own support/query split.

The challenge: The model must generalize from very few examples (often just 1 per
category) to correctly classify new words it hasn't seen in the support set.

Data flow per episode:
  1. Randomly pick N categories (e.g., animals, furniture)
  2. For each category, sample K words for the support set and Q words for the query set
  3. The model first processes the support set (words + their labels) to adapt its
     internal representations via plastic synapses
  4. Then the model classifies the query words, and loss is computed on these predictions
"""
import random
from dataclasses import dataclass
from typing import Literal


# =============================================================================
# SEMANTIC CATEGORIES VOCABULARY
# =============================================================================
# Each category contains 10 words. During training/testing, these are split
# 50/50 so that the support and query sets draw from the same pool of 5 words
# per category, ensuring the model must generalize across words within a category.
CATEGORIES = {
    "animals": ["dog", "cat", "fish", "mouse", "deer", "bear", "wolf", "lion", "tiger", "elephant"],
    "clothing": ["shirt", "jacket", "skirt", "dress", "pants", "shorts", "shoes", "hat", "coat", "gloves"],
    "furniture": ["table", "chair", "carpet", "couch", "bed", "desk", "shelf", "cabinet", "wardrobe", "dresser"],
}

# Flatten all words into a single list and build a reverse lookup.
# ALL_WORDS is used by encoders to build their vocabularies.
# WORD_TO_CATEGORY maps each word back to its semantic category.
ALL_WORDS = []
WORD_TO_CATEGORY = {}
for category, words in CATEGORIES.items():
    ALL_WORDS.extend(words)
    for word in words:
        WORD_TO_CATEGORY[word] = category


@dataclass
class WordCategoryConfig:
    """
    Configuration for word category classification task.
    
    This controls the structure of each few-shot episode:
      - ways: How many categories to classify between (N in "N-way")
      - shots: How many labeled examples per category the model sees (K in "K-shot")
      - queries: How many unlabeled examples per category the model must classify
      - split: Whether to use the train or test portion of each category's words
    
    Example: With ways=3, shots=1, queries=2, each episode has:
      - 3 categories randomly chosen (e.g., animals, clothing, furniture)
      - 3 support examples (1 per category, with labels)
      - 6 query examples (2 per category, to be classified)
    """
    ways: int = 3       # Number of categories per episode (N-way)
    shots: int = 1      # Number of labeled support examples per category (K-shot)
    queries: int = 15   # Number of query examples per category to classify
    split: Literal["train", "test"] = "train"  # Which half of the vocabulary to use
    
    def __post_init__(self):
        """Validate configuration and compute the train/test vocabulary split."""
        if self.ways > len(CATEGORIES):
            raise ValueError(f"ways ({self.ways}) cannot exceed number of categories ({len(CATEGORIES)})")
        
        # Split each category's vocabulary into disjoint train/test halves.
        # This ensures the model is evaluated on words it never saw during training,
        # testing true generalization of category understanding rather than memorization.
        # With 10 words per category, each split gets 5 words.
        self.category_splits = {}
        for category, words in CATEGORIES.items():
            n_words = len(words)
            train_size = n_words // 2  # 50/50 split: 5 train + 5 test per category
            
            if self.split == "train":
                self.category_splits[category] = words[:train_size]
            else:
                self.category_splits[category] = words[train_size:]


class WordCategoryTask:
    """
    Few-shot word category classification task.
    
    This is the core task class that generates episodes for meta-learning.
    Each episode is a self-contained few-shot learning problem:
    
      1. Randomly select `ways` categories for this episode
      2. From each category, sample `shots` words for the support set (labeled)
         and `queries` words for the query set (to be predicted)
      3. The support set teaches the model which words belong to which category
      4. The query set tests whether the model can generalize to other words
    
    The model never sees the query labels during forward pass -- those are only
    used to compute the loss and accuracy.
    
    Important: Support and query words are drawn WITHOUT replacement from the
    same pool, so they are always different words. This forces the model to
    learn the semantic category rather than just memorizing specific words.
    """
    
    def __init__(self, config: WordCategoryConfig):
        self.config = config
        self.available_categories = list(CATEGORIES.keys())
        
        # Validate that each category has enough words for both support and query sets.
        # We need at least (shots + queries) words available in the current split.
        for category, words in config.category_splits.items():
            min_needed = config.shots + config.queries
            if len(words) < min_needed:
                raise ValueError(
                    f"Category '{category}' only has {len(words)} words in {config.split} split, "
                    f"but needs at least {min_needed} (shots={config.shots} + queries={config.queries})"
                )
    
    def sample_episode(self, seed: int | None = None):
        """
        Sample a single episode for few-shot learning.
        
        This is the main method that constructs one complete few-shot learning trial.
        The returned data is structured so the model can:
          1. First process support_strings with their support_labels to adapt
          2. Then classify query_strings and be evaluated against query_labels
        
        Args:
            seed: Random seed for reproducibility (optional)
            
        Returns:
            Tuple containing:
            - support_strings: List of labeled support words (length: ways * shots)
            - support_labels: Integer class labels 0..ways-1 for each support word
            - query_strings: List of query words to classify (length: ways * queries)
            - query_labels: True integer class labels 0..ways-1 for evaluation
            - category_names: List of category name strings for this episode
            
        Example output (3-way, 1-shot, 2-query):
            support_strings = ["chair", "dog", "shirt"]
            support_labels  = [1, 0, 2]           # furniture=1, animals=0, clothing=2
            query_strings   = ["bed", "cat", "pants", "desk", "hat", "wolf"]
            query_labels    = [1, 0, 2, 1, 2, 0]
            category_names  = ["animals", "furniture", "clothing"]
        """
        if seed is not None:
            random.seed(seed)
        
        # Step 1: Randomly pick which categories appear in this episode.
        # This means each episode tests different combinations of categories.
        episode_categories = random.sample(self.available_categories, self.config.ways)
        
        # Step 2: For each selected category, sample words for support and query sets.
        support_strings = []
        support_labels = []
        query_strings = []
        query_labels = []
        
        for class_idx, category in enumerate(episode_categories):
            # Get the word pool for this category (already split into train/test)
            available_words = self.config.category_splits[category]
            
            # Sample words WITHOUT replacement -- support and query words are disjoint.
            # This is critical: the model can't just memorize support words,
            # it must learn the underlying category structure.
            sampled_words = random.sample(available_words, self.config.shots + self.config.queries)
            
            # First `shots` words become the labeled support set
            support_words = sampled_words[:self.config.shots]
            for word in support_words:
                support_strings.append(word)
                support_labels.append(class_idx)  # class_idx is 0..ways-1
            
            # Remaining words become the query set (labels used only for evaluation)
            query_words = sampled_words[self.config.shots:]
            for word in query_words:
                query_strings.append(word)
                query_labels.append(class_idx)
        
        # Step 3: Shuffle both sets independently so the model can't exploit ordering.
        # Without shuffling, the model could learn that the first support example
        # is always class 0, etc.
        support_indices = list(range(len(support_strings)))
        random.shuffle(support_indices)
        support_strings = [support_strings[i] for i in support_indices]
        support_labels = [support_labels[i] for i in support_indices]
        
        query_indices = list(range(len(query_strings)))
        random.shuffle(query_indices)
        query_strings = [query_strings[i] for i in query_indices]
        query_labels = [query_labels[i] for i in query_indices]
        
        return support_strings, support_labels, query_strings, query_labels, episode_categories
    
    def get_split_info(self) -> dict:
        """
        Get information about the train/test split.
        
        Returns:
            Dictionary with split information
        """
        info = {
            "split": self.config.split,
            "total_categories": len(CATEGORIES),
            "categories": {},
        }
        
        for category, words in self.config.category_splits.items():
            info["categories"][category] = {
                "available_words": len(words),
                "words": words,
            }
        
        return info


if __name__ == "__main__":
    # Test the task
    print("=== Testing Word Category Classification Task ===\n")
    
    config = WordCategoryConfig(
        ways=3,
        shots=1,
        queries=2,  # Only 2 queries since we have 3 words per category in train split
        split="train",
    )
    
    task = WordCategoryTask(config)
    print(f"Split info: {task.get_split_info()}\n")
    
    # Sample an episode
    support_strings, support_labels, query_strings, query_labels, category_names = task.sample_episode(seed=42)
    
    print(f"Episode categories: {category_names}\n")
    print("Support set:")
    for string, label in zip(support_strings, support_labels):
        print(f"  '{string}' -> class {label} ({category_names[label]})")
    
    print("\nQuery set:")
    for string, label in zip(query_strings, query_labels):
        print(f"  '{string}' -> class {label} ({category_names[label]})")
    
    print("\n" + "="*60)
    
    # Test train/test split
    print("\nTesting train/test split:")
    train_task = WordCategoryTask(WordCategoryConfig(ways=3, shots=1, queries=2, split="train"))
    test_task = WordCategoryTask(WordCategoryConfig(ways=3, shots=1, queries=1, split="test"))  # Only 1 query for test since we have 3 words
    
    print("\nTrain split:")
    for category, info in train_task.get_split_info()["categories"].items():
        print(f"  {category}: {info['words']}")
    
    print("\nTest split:")
    for category, info in test_task.get_split_info()["categories"].items():
        print(f"  {category}: {info['words']}")
