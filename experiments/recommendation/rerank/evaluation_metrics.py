"""
Evaluation metrics for ranking quality assessment.

This module provides standard IR metrics:
- Recall@k: Fraction of relevant items found in top-k
- NDCG@k: Normalized Discounted Cumulative Gain at k (using sklearn)
"""

import numpy as np
from typing import List, Dict, Set
from sklearn.metrics import ndcg_score
from data_schema import FeedbackItem


def recall_at_k(ranked_ids: List[str], positive_ids: Set[str], k: int) -> float:
    """
    Calculate Recall@k: fraction of positive items found in top-k ranked results.

    Recall@k = |positive_ids ∩ top_k_ranked| / |positive_ids|

    Args:
        ranked_ids: List of item IDs in ranked order (best first)
        positive_ids: Set of positive/relevant item IDs
        k: Number of top items to consider

    Returns:
        Recall@k value between 0.0 and 1.0
        Returns 0.0 if there are no positive items
    """
    if not positive_ids:
        return 0.0

    top_k = set(ranked_ids[:k])
    found = len(top_k.intersection(positive_ids))

    return found / len(positive_ids)


def ndcg_at_k(ranked_ids: List[str], labels: Dict[str, int], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at k using sklearn.

    Args:
        ranked_ids: List of item IDs in ranked order (best first)
        labels: Dictionary mapping item IDs to relevance scores (+1, 0, -1)
        k: Number of top items to consider

    Returns:
        NDCG@k value between 0.0 and 1.0
        Returns 0.0 if there are no labeled items
    """
    if not labels or not ranked_ids:
        return 0.0

    # Convert labels to relevance scores for NDCG
    # +1 (liked) -> 2, 0 (viewed) -> 1, -1 (disliked) -> 0
    relevance_map = {1: 2, 0: 1, -1: 0}

    # Build true relevance vector in the order of ranked_ids
    y_true = []
    y_score = []

    for i, item_id in enumerate(ranked_ids[:k]):
        if item_id in labels:
            relevance = relevance_map[labels[item_id]]
            y_true.append(relevance)
            # Score is based on position (higher position = higher score)
            y_score.append(k - i)
        else:
            y_true.append(0)
            y_score.append(k - i)

    # Need at least one relevant item
    if sum(y_true) == 0:
        return 0.0

    # sklearn expects 2D arrays
    y_true = np.array([y_true])
    y_score = np.array([y_score])

    return ndcg_score(y_true, y_score, k=k)


def evaluate_ranking(
    ranked_ids: List[str],
    feedback_item: FeedbackItem,
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Evaluate a ranking using multiple metrics at different k values.

    Args:
        ranked_ids: List of item IDs in ranked order (best first)
        feedback_item: FeedbackItem containing labels
        k_values: List of k values to evaluate at

    Returns:
        Dictionary with metric names as keys and values as floats
        Example: {'recall@1': 0.0, 'recall@3': 0.5, 'ndcg@1': 0.0, 'ndcg@3': 0.63, ...}
    """
    results = {}

    # Get positive items for recall calculation
    positive_ids = set(feedback_item.get_positive_papers())
    labels = feedback_item.labels

    # Metadata
    results['total_candidates'] = len(ranked_ids)
    results['num_positives'] = len(positive_ids)
    results['num_labeled'] = len(labels)

    # Calculate metrics for each k
    for k in k_values:
        results[f'recall@{k}'] = recall_at_k(ranked_ids, positive_ids, k)
        results[f'ndcg@{k}'] = ndcg_at_k(ranked_ids, labels, k)

    return results


def evaluate_user_split(
    feedback_items: List[FeedbackItem],
    get_ranking_fn,
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, any]:
    """
    Evaluate rankings for a list of feedback items (e.g., a user's dev set).

    Args:
        feedback_items: List of FeedbackItem objects to evaluate
        get_ranking_fn: Function that takes a FeedbackItem and returns ranked list of IDs
        k_values: List of k values to evaluate at

    Returns:
        Dictionary containing:
        - 'per_query': List of per-query results
        - 'average': Averaged metrics across all queries
    """
    per_query_results = []

    for item in feedback_items:
        # Get ranking for this query
        ranked_ids = get_ranking_fn(item)

        # Evaluate this ranking
        results = evaluate_ranking(ranked_ids, item, k_values)
        results['query'] = item.query_context[:100] + "..."  # Add truncated query for reference
        results['timestamp'] = str(item.timestamp)

        per_query_results.append(results)

    # Calculate averages
    if not per_query_results:
        return {'per_query': [], 'average': {}}

    average = {}
    metric_keys = [key for key in per_query_results[0].keys()
                   if key not in ['query', 'timestamp', 'total_candidates', 'num_positives', 'num_labeled']]

    for key in metric_keys:
        values = [r[key] for r in per_query_results]
        average[key] = sum(values) / len(values)

    # Add total counts
    average['total_queries'] = len(per_query_results)
    average['total_candidates'] = sum(r['total_candidates'] for r in per_query_results)
    average['total_positives'] = sum(r['num_positives'] for r in per_query_results)
    average['total_labeled'] = sum(r['num_labeled'] for r in per_query_results)

    return {
        'per_query': per_query_results,
        'average': average
    }


def print_evaluation_results(results: Dict[str, any], title: str = "Evaluation Results"):
    """
    Pretty print evaluation results.

    Args:
        results: Results dictionary from evaluate_user_split
        title: Title for the output
    """
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")

    avg = results['average']

    print(f"\nOverall Statistics:")
    print(f"  Total queries: {avg.get('total_queries', 0)}")
    print(f"  Total candidates: {avg.get('total_candidates', 0)}")
    print(f"  Total positives: {avg.get('total_positives', 0)}")
    print(f"  Total labeled: {avg.get('total_labeled', 0)}")

    print(f"\nAverage Metrics:")

    # Extract k values from metric names
    k_values = sorted(set(int(k.split('@')[1]) for k in avg.keys() if '@' in k))

    print(f"\n  {'Metric':<15} " + " ".join(f"k={k:<3}" for k in k_values))
    print(f"  {'-'*15} " + " ".join(f"{'-'*6}" for _ in k_values))

    for metric_type in ['recall', 'ndcg']:
        values = [f"{avg.get(f'{metric_type}@{k}', 0):.4f}" for k in k_values]
        print(f"  {metric_type:<15} " + " ".join(f"{v:<6}" for v in values))

    print(f"\n{'='*80}\n")
