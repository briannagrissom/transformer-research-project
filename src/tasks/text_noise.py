"""
Utility functions for adding noise to text strings.
Implements character substitution and adjacent character transposition.
"""
import random
import string
from typing import Literal


def add_substitution_noise(text: str, noise_prob: float, seed: int | None = None) -> str:
    """
    Add random character substitution noise to a string.
    
    Each character has a probability `noise_prob` of being replaced with a random
    lowercase letter.
    
    Args:
        text: Input string to corrupt
        noise_prob: Probability (0.0 to 1.0) that each character is substituted
        seed: Random seed for reproducibility (optional)
        
    Returns:
        Corrupted string with substituted characters
        
    Example:
        >>> add_substitution_noise("hello", 0.2, seed=42)
        "hellp"  # 'o' substituted with 'p'
    """
    if seed is not None:
        random.seed(seed)
    
    if noise_prob <= 0.0:
        return text
    
    chars = list(text.lower())
    substitution_chars = string.ascii_lowercase
    
    for i in range(len(chars)):
        # Only substitute alphabetic characters
        if chars[i].isalpha() and random.random() < noise_prob:
            # Pick a random lowercase letter (could be same as original)
            chars[i] = random.choice(substitution_chars)
    
    return ''.join(chars)


def add_transposition_noise(text: str, noise_prob: float, seed: int | None = None) -> str:
    """
    Add adjacent character transposition (swap) noise to a string.
    
    Each adjacent pair of characters has a probability `noise_prob` of being swapped.
    To avoid double-swapping, we only consider non-overlapping pairs.
    
    Args:
        text: Input string to corrupt
        noise_prob: Probability (0.0 to 1.0) that each pair is transposed
        seed: Random seed for reproducibility (optional)
        
    Returns:
        Corrupted string with transposed characters
        
    Example:
        >>> add_transposition_noise("hello", 0.5, seed=42)
        "helol"  # last two characters 'l' and 'o' swapped
    """
    if seed is not None:
        random.seed(seed)
    
    if noise_prob <= 0.0 or len(text) < 2:
        return text
    
    chars = list(text.lower())
    i = 0
    
    while i < len(chars) - 1:
        # Swap adjacent characters with probability noise_prob
        if random.random() < noise_prob:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            i += 2  # Skip the next character to avoid double-swapping
        else:
            i += 1
    
    return ''.join(chars)


def add_noise(
    text: str,
    noise_type: Literal["substitution", "transposition"],
    noise_prob: float,
    seed: int | None = None,
) -> str:
    """
    Add specified type of noise to text.
    
    Args:
        text: Input string to corrupt
        noise_type: Type of noise - "substitution" or "transposition"
        noise_prob: Probability (0.0 to 1.0) of noise application
        seed: Random seed for reproducibility (optional)
        
    Returns:
        Corrupted string
        
    Raises:
        ValueError: If noise_type is not recognized
    """
    if noise_type == "substitution":
        return add_substitution_noise(text, noise_prob, seed)
    elif noise_type == "transposition":
        return add_transposition_noise(text, noise_prob, seed)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")


def compute_edit_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein edit distance between two strings.
    
    This is the minimum number of single-character edits (insertions, deletions,
    or substitutions) needed to transform s1 into s2.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Edit distance (non-negative integer)
    """
    m, n = len(s1), len(s2)
    
    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1]   # substitution
                )
    
    return dp[m][n]


def add_single_char_noise(text: str, noise_type: Literal["substitution", "transposition"] = "substitution") -> str:
    """
    Add noise to exactly ONE character in the string.
    
    For substitution: randomly pick one character and replace it with a different letter.
    For transposition: randomly pick one adjacent pair and swap them.
    
    Args:
        text: Input string to corrupt
        noise_type: Type of noise - "substitution" or "transposition"
        
    Returns:
        Corrupted string with exactly one character changed
        
    Example:
        >>> add_single_char_noise("hello", "substitution")
        "hello" -> "hxllo" (random position, random letter)
    """
    if len(text) == 0:
        return text
    
    text_lower = text.lower()
    chars = list(text_lower)
    
    if noise_type == "substitution":
        # Pick a random position
        pos = random.randint(0, len(chars) - 1)
        
        # If it's not alphabetic, just return original
        if not chars[pos].isalpha():
            return text_lower
        
        # Pick a random letter different from the current one
        original_char = chars[pos]
        possible_chars = [c for c in string.ascii_lowercase if c != original_char]
        chars[pos] = random.choice(possible_chars)
        
    elif noise_type == "transposition":
        # Need at least 2 characters to transpose
        if len(chars) < 2:
            return text_lower
        
        # Pick a random position to swap with next position
        pos = random.randint(0, len(chars) - 2)
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
    
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")
    
    return ''.join(chars)


def compute_character_error_rate(original: str, noisy: str) -> float:
    """
    Compute character error rate (CER) between original and noisy strings.
    
    CER = edit_distance / len(original)
    
    Args:
        original: Original clean string
        noisy: Corrupted string
        
    Returns:
        Character error rate (0.0 to infinity, typically 0.0 to 1.0)
    """
    if len(original) == 0:
        return 0.0 if len(noisy) == 0 else float('inf')
    
    distance = compute_edit_distance(original, noisy)
    return distance / len(original)
