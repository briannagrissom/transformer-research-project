"""
Few-shot word category classification task.
Classifies words into semantic categories (animals, clothing, furniture, etc.)
"""
import random
from dataclasses import dataclass
from typing import Literal


# Define semantic categories with their vocabulary
CATEGORIES = {
    "animals": ["dog", "cat", "fish", "mouse", "deer", "bear", "wolf", "lion", "tiger", "elephant"],
    "clothing": ["shirt", "jacket", "skirt", "dress", "pants", "shorts", "shoes", "hat", "coat", "gloves"],
    "furniture": ["table", "chair", "carpet", "couch", "bed", "desk", "shelf", "cabinet", "wardrobe", "dresser"],
}

# Flatten all words for easy access
ALL_WORDS = []
WORD_TO_CATEGORY = {}
for category, words in CATEGORIES.items():
    ALL_WORDS.extend(words)
    for word in words:
        WORD_TO_CATEGORY[word] = category


@dataclass
class WordCategoryConfig:
    """Configuration for word category classification task."""
    ways: int = 3  # Number of categories per episode
    shots: int = 1  # Number of support examples per category
    queries: int = 15  # Number of query examples per category
    split: Literal["train", "test"] = "train"
    
    def __post_init__(self):
        """Validate configuration."""
        if self.ways > len(CATEGORIES):
            raise ValueError(f"ways ({self.ways}) cannot exceed number of categories ({len(CATEGORIES)})")
        
        # Calculate train/test split for each category
        self.category_splits = {}
        for category, words in CATEGORIES.items():
            n_words = len(words)
            # Use 50/50 split since we only have 10 words per category
            # This gives 5 words for train and 5 for test
            train_size = n_words // 2
            
            if self.split == "train":
                self.category_splits[category] = words[:train_size]
            else:
                self.category_splits[category] = words[train_size:]


class WordCategoryTask:
    """
    Few-shot word category classification task.
    
    Each episode samples `ways` categories, then samples support and query words
    from each category.
    """
    
    def __init__(self, config: WordCategoryConfig):
        self.config = config
        self.available_categories = list(CATEGORIES.keys())
        
        # Check if we have enough words in each category for the split
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
        
        Args:
            seed: Random seed for reproducibility (optional)
            
        Returns:
            Tuple containing:
            - support_strings: List of support words
            - support_labels: List of integer labels (0 to ways-1) for support
            - query_strings: List of query words
            - query_labels: List of integer labels (0 to ways-1) for query
            - category_names: List of category names for this episode
        """
        if seed is not None:
            random.seed(seed)
        
        # Sample `ways` categories for this episode
        episode_categories = random.sample(self.available_categories, self.config.ways)
        
        # Generate support and query sets
        support_strings = []
        support_labels = []
        query_strings = []
        query_labels = []
        
        for class_idx, category in enumerate(episode_categories):
            # Get available words for this category in current split
            available_words = self.config.category_splits[category]
            
            # Sample words for support and query (without replacement within episode)
            sampled_words = random.sample(available_words, self.config.shots + self.config.queries)
            
            # First `shots` words go to support
            support_words = sampled_words[:self.config.shots]
            for word in support_words:
                support_strings.append(word)
                support_labels.append(class_idx)
            
            # Remaining words go to query
            query_words = sampled_words[self.config.shots:]
            for word in query_words:
                query_strings.append(word)
                query_labels.append(class_idx)
        
        # Shuffle support and query sets independently
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
