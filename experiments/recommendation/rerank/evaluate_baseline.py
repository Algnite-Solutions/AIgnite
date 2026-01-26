"""
Baseline evaluation script.

Evaluates random shuffle baseline on dev sets.
"""

import json
import random
from collections import defaultdict

from data_schema import UserTimeline, FeedbackItem
from chronological_splitter import split_multiple_users
from evaluation_metrics import evaluate_user_split, print_evaluation_results


def baseline_ranking(feedback_item: FeedbackItem, seed: int = 42) -> list:
    """
    Baseline ranking: random shuffle of candidate_set.

    Args:
        feedback_item: FeedbackItem with candidate_set
        seed: Random seed for reproducibility

    Returns:
        List of paper IDs in random order
    """
    candidates = feedback_item.candidate_set.copy()
    random.seed(seed)
    random.shuffle(candidates)
    return candidates


def main():
    # Load the user feedback data
    data_file = "/Users/bran/Desktop/AIgnite-Solutions/AIgnite/experiments/data/result_1214/export_user_retrieve_results.jsonl"

    print("="*80)
    print("BASELINE EVALUATION - Random Shuffle")
    print("="*80)
    print(f"\nData file: {data_file}\n")

    # Group entries by user
    user_entries = defaultdict(list)

    with open(data_file) as IN:
        for line in IN:
            entry = json.loads(line)
            user_entries[entry['user_name']].append(entry)

    print(f"Found {len(user_entries)} unique users\n")

    # Create UserTimeline objects
    user_timelines = []
    for user_name, entries in user_entries.items():
        try:
            timeline = UserTimeline.from_jsonl_entries(entries)
            user_timelines.append(timeline)
        except Exception as e:
            print(f"Warning: Failed to create timeline for {user_name}: {e}")

    # Split data chronologically
    print("Splitting data chronologically (80/20 train/dev)...\n")
    user_splits = split_multiple_users(user_timelines, train_ratio=0.8)

    print(f"Successfully split data for {len(user_splits)} users\n")

    # Filter to only users with positive feedbacks
    users_with_positives = []
    for timeline in user_timelines:
        if len(timeline.get_all_positive_papers()) > 0 and timeline.user_name in user_splits:
            users_with_positives.append(timeline.user_name)

    print(f"Users with positive feedbacks: {users_with_positives}")
    print(f"Number of users to evaluate: {len(users_with_positives)}\n")

    # Evaluate baseline for each user
    # Merge all dev queries into one per user
    all_results = {}

    for user_name, split in user_splits.items():
        # Skip users without positive feedbacks
        if user_name not in users_with_positives:
            print(f"\nSkipping {user_name} (no positive feedbacks)")
            continue
        print(f"\n{'='*80}")
        print(f"Evaluating: {user_name}")
        print(f"{'='*80}")
        print(f"Dev set size: {len(split.dev)} queries (will be merged into 1)")

        # Merge all dev queries into a single candidate pool
        merged_candidates = []
        merged_labels = {}

        for item in split.dev:
            # Add all candidates from this query
            merged_candidates.extend(item.candidate_set)
            # Merge labels
            merged_labels.update(item.labels)

        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for cand in merged_candidates:
            if cand not in seen:
                seen.add(cand)
                unique_candidates.append(cand)

        # Create a merged FeedbackItem for evaluation
        merged_item = FeedbackItem(
            timestamp=split.dev[-1].timestamp,  # Use latest timestamp
            user_name=user_name,
            query_context=f"Merged dev set for {user_name}",
            candidate_set=unique_candidates,
            labels=merged_labels,
            search_strategy="merged",
            top_k_ids=unique_candidates
        )

        print(f"  Total unique candidates: {len(unique_candidates)}")
        print(f"  Total labeled papers: {len(merged_labels)}")
        print(f"  Positive papers: {len(merged_item.get_positive_papers())}")
        print(f"  Negative papers: {len(merged_item.get_negative_papers())}")

        # Evaluate using baseline ranking on merged item
        results = evaluate_user_split(
            feedback_items=[merged_item],
            get_ranking_fn=baseline_ranking,
            k_values=[1, 3, 5, 10, 15, 20]
        )

        all_results[user_name] = results

        # Print results for this user
        print_evaluation_results(results, title=f"Baseline Results - {user_name}")

    # Overall statistics across all users
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS - ALL USERS")
    print(f"{'='*80}\n")

    total_queries = sum(r['average']['total_queries'] for r in all_results.values())
    total_candidates = sum(r['average']['total_candidates'] for r in all_results.values())
    total_positives = sum(r['average']['total_positives'] for r in all_results.values())
    total_labeled = sum(r['average']['total_labeled'] for r in all_results.values())

    print(f"Total users: {len(all_results)}")
    print(f"Total queries: {total_queries}")
    print(f"Total candidates: {total_candidates}")
    print(f"Total positives: {total_positives}")
    print(f"Total labeled: {total_labeled}")
    print(f"Average positives per query: {total_positives/total_queries:.2f}")
    print(f"Average labeled per query: {total_labeled/total_queries:.2f}")

    # Macro-averaged metrics (average across users)
    print(f"\nMacro-Averaged Metrics (average across users):")
    k_values = [1, 3, 5, 10, 15, 20]

    print(f"\n  {'Metric':<15} " + " ".join(f"k={k:<3}" for k in k_values))
    print(f"  {'-'*15} " + " ".join(f"{'-'*6}" for _ in k_values))

    for metric_type in ['recall', 'ndcg']:
        values = []
        for k in k_values:
            metric_name = f'{metric_type}@{k}'
            avg_value = sum(r['average'][metric_name] for r in all_results.values()) / len(all_results)
            values.append(f"{avg_value:.4f}")
        print(f"  {metric_type:<15} " + " ".join(f"{v:<6}" for v in values))

    print(f"\n{'='*80}\n")

    # Save results to file
    output_file = "baseline_evaluation_results.json"
    print(f"Saving detailed results to {output_file}...")


    with open(output_file, 'w') as f:
        # Convert results to JSON-serializable format
        json_results = {}
        for user_name, result in all_results.items():
            json_results[user_name] = {
                'average': result['average'],
                'per_query': result['per_query']
            }
        json.dump(json_results, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
