"""
Few-shot text classification dataset with noise.
Implements episode sampling for denoising text classification task.
"""
import random
from dataclasses import dataclass
from typing import Literal

from src.tasks.text_noise import add_noise


# Vocabulary of 100 common English words
VOCABULARY = [
    "apple", "banana", "orange", "grape", "cherry",
    "strawberry", "blueberry", "raspberry", "blackberry", "watermelon",
    "pineapple", "mango", "papaya", "peach", "plum",
    "table", "chair", "desk", "bed", "couch",
    "lamp", "door", "window", "mirror", "clock",
    "book", "pencil", "paper", "notebook", "eraser",
    "computer", "keyboard", "mouse", "monitor", "printer",
    "phone", "camera", "speaker", "headphone", "microphone",
    "car", "bus", "train", "plane", "boat",
    "bicycle", "motorcycle", "truck", "helicopter", "rocket",
    "dog", "cat", "bird", "fish", "rabbit",
    "horse", "elephant", "lion", "tiger", "bear",
    "monkey", "zebra", "giraffe", "penguin", "dolphin",
    "mountain", "river", "ocean", "forest", "desert",
    "island", "valley", "lake", "beach", "volcano",
    "house", "building", "castle", "bridge", "tower",
    "school", "hospital", "library", "museum", "theater",
    "guitar", "piano", "violin", "drum", "flute",
    "trumpet", "saxophone", "clarinet", "harp", "cello",
    "summer", "winter", "spring", "autumn", "morning",
    "evening", "midnight", "sunrise", "sunset", "rainbow",
]


@dataclass
class TextClassificationConfig:
    """Configuration for text classification episodes."""
    
    ways: int = 5  # Number of classes per episode
    shots: int = 1  # Number of support examples per class
    queries: int = 15  # Number of query examples per class
    noise_type: Literal["substitution", "transposition"] = "substitution"
    noise_prob: float = 0.1  # Probability of noise per character
    split: Literal["train", "test"] = "train"
    vocab_size: int = 100  # Total vocabulary size
    
    def __post_init__(self):
        """Validate configuration."""
        if self.ways > self.vocab_size:
            raise ValueError(f"ways ({self.ways}) cannot exceed vocab_size ({self.vocab_size})")
        
        if self.split == "train":
            # Use first 80% of vocabulary for training
            self.class_indices = list(range(int(0.8 * self.vocab_size))) # 0, 1, ..., 79
        else:
            # Use last 20% of vocabulary for testing (held-out classes)
            self.class_indices = list(range(int(0.8 * self.vocab_size), self.vocab_size)) # 80, 81, ..., 99
        
        if self.ways > len(self.class_indices):
            raise ValueError(
                f"ways ({self.ways}) exceeds available classes "
                f"({len(self.class_indices)}) for split '{self.split}'"
            )


class TextClassificationTask:
    """
    Few-shot text classification task with denoising.
    
    Each episode samples `ways` classes from the vocabulary, provides `shots` noisy
    support examples per class, and tests on `queries` clean query examples per class.
    """
    
    def __init__(self, config: TextClassificationConfig):
        """
        Args:
            config: Configuration for the task
        """
        self.config = config
        self.vocabulary = VOCABULARY[:config.vocab_size]
        
        # Validate that we have enough vocabulary
        if len(self.vocabulary) < config.vocab_size:
            raise ValueError(
                f"Requested vocab_size {config.vocab_size} but only "
                f"{len(self.vocabulary)} words available"
            )
    
    def sample_episode(self, seed: int | None = None):
        """
        Sample a single episode for few-shot learning.
        
        Args:
            seed: Random seed for reproducibility (optional)
            
        Returns:
            Tuple containing:
            - support_strings: List of noisy support strings
            - support_labels: List of integer labels (0 to ways-1) for support strings
            - query_strings: List of clean query strings  
            - query_labels: List of integer labels (0 to ways-1) for query strings
            - class_words: List of the actual words selected for this episode
        """
        if seed is not None:
            random.seed(seed)
        
        # Sample `ways` classes from available class indices
        episode_class_indices = random.sample(self.config.class_indices, self.config.ways)
        class_words = [self.vocabulary[idx] for idx in episode_class_indices] # length: ways
        
        # Generate support set (noisy)
        support_strings = [] # length: ways * shots
        support_labels = [] # length: ways * shots
        
        for class_idx, word in enumerate(class_words):
            for _ in range(self.config.shots):
                # Add noise to the support example
                noisy_word = add_noise(
                    word,
                    noise_type=self.config.noise_type,
                    noise_prob=self.config.noise_prob,
                )
                support_strings.append(noisy_word)
                support_labels.append(class_idx)
        
        # Generate query set (clean)
        query_strings = [] # length: ways * queries
        query_labels = [] # length: ways * queries
        
        for class_idx, word in enumerate(class_words):
            for _ in range(self.config.queries):
                # Query examples are always clean
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
        
        return support_strings, support_labels, query_strings, query_labels, class_words
    
    def get_vocab_split_info(self) -> dict:
        """
        Get information about the train/test vocabulary split.
        
        Returns:
            Dictionary with split information
        """
        train_size = int(0.8 * self.config.vocab_size)
        test_size = self.config.vocab_size - train_size
        
        return {
            "total_vocab_size": self.config.vocab_size,
            "train_classes": train_size,
            "test_classes": test_size,
            "split": self.config.split,
            "available_classes": len(self.config.class_indices),
        }


def test_noise_levels():
    """Test function to visualize noise at different levels."""
    test_word = "snowball"
    noise_probs = [0.0, 0.05, 0.1, 0.2]
    
    print("Testing substitution noise:")
    for prob in noise_probs:
        noisy = add_noise(test_word, "substitution", prob, seed=42)
        print(f"  p={prob:.2f}: '{test_word}' -> '{noisy}'")
    
    print("\nTesting transposition noise:")
    for prob in noise_probs:
        noisy = add_noise(test_word, "transposition", prob, seed=42)
        print(f"  p={prob:.2f}: '{test_word}' -> '{noisy}'")


if __name__ == "__main__":
    # Test the task
    print("=== Testing Text Classification Task ===\n")
    
    config = TextClassificationConfig(
        ways=5,
        shots=1,
        queries=3,
        noise_type="substitution",
        noise_prob=0.1,
        split="train",
    )
    
    task = TextClassificationTask(config)
    print(f"Vocabulary split info: {task.get_vocab_split_info()}\n")
    
    # Sample an episode
    support_strings, support_labels, query_strings, query_labels, class_words = task.sample_episode(seed=42)
    
    print(f"Episode classes: {class_words}\n")
    print("Support set (noisy):")
    for string, label in zip(support_strings, support_labels):
        print(f"  '{string}' -> class {label} ({class_words[label]})")
    
    print("\nQuery set (clean):")
    for string, label in zip(query_strings[:5], query_labels[:5]):  # Show first 5
        print(f"  '{string}' -> class {label}")
    
    print(f"\n... and {len(query_strings) - 5} more query examples")
    
    print("\n" + "="*50)
    test_noise_levels()
