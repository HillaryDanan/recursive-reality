"""
Metrics for measuring information degradation and accuracy
Based on established NLP evaluation methods
"""

from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import Levenshtein
from loguru import logger

def calculate_similarity(text1: str, text2: str, method: str = 'cosine') -> float:
    """
    Calculate similarity between two texts
    Methods from Zhang et al. (2019) on text similarity metrics
    """
    if method == 'cosine':
        # TF-IDF based cosine similarity
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0
            
    elif method == 'levenshtein':
        # Normalized Levenshtein distance
        distance = Levenshtein.distance(text1, text2)
        max_len = max(len(text1), len(text2))
        return 1.0 - (distance / max_len) if max_len > 0 else 0.0
        
    elif method == 'jaccard':
        # Jaccard similarity on word sets
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if len(union) > 0 else 0.0
    
    else:
        raise ValueError(f"Unknown method: {method}")

def calculate_accuracy(generated: str, ground_truth: str, key_terms: List[str] = None) -> float:
    """
    Calculate factual accuracy score
    Based on BERTScore methodology (Zhang et al., 2020) simplified for interpretability
    """
    score = 0.0
    
    # Basic similarity component (50% weight)
    similarity = calculate_similarity(generated, ground_truth, 'cosine')
    score += similarity * 0.5
    
    # Key terms preservation (50% weight) if provided
    if key_terms:
        generated_lower = generated.lower()
        preserved = sum(1 for term in key_terms if term.lower() in generated_lower)
        preservation_rate = preserved / len(key_terms) if key_terms else 0
        score += preservation_rate * 0.5
    else:
        # If no key terms, use Jaccard similarity for second component
        jaccard = calculate_similarity(generated, ground_truth, 'jaccard')
        score += jaccard * 0.5
    
    return min(score, 1.0)  # Cap at 1.0

def calculate_degradation_rate(similarities: List[float]) -> Tuple[float, float]:
    """
    Calculate degradation rate across layers
    Returns: (mean_degradation_per_layer, exponential_decay_coefficient)
    Following information theory (Shannon, 1948)
    """
    if len(similarities) < 2:
        return 0.0, 0.0
    
    # Linear degradation rate
    degradations = [similarities[i] - similarities[i+1] for i in range(len(similarities)-1)]
    mean_degradation = np.mean(degradations)
    
    # Fit exponential decay: y = a * exp(-b * x)
    # Using log transform: log(y) = log(a) - b * x
    try:
        x = np.arange(len(similarities))
        log_similarities = np.log(np.clip(similarities, 1e-10, 1))  # Avoid log(0)
        
        # Linear regression on log-transformed data
        coeffs = np.polyfit(x, log_similarities, 1)
        decay_coefficient = -coeffs[0]  # The slope is -b
        
        return mean_degradation, decay_coefficient
    except:
        return mean_degradation, 0.0

logger.info("Metrics module loaded - using established NLP evaluation methods")
