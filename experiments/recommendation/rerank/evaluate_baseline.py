"""
Baseline evaluation script.

Evaluates random shuffle baseline on dev sets.
Supports both file-based (legacy) and database-backed data fetching.
"""

import json
import random
import logging
import os
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from data_schema import UserTimeline, FeedbackItem, ThreeWayDataSplit
from data_fetcher import UserDataFetcher, split_user_timelines_three_way, print_split_summary
from evaluation_metrics import evaluate_user_split, print_evaluation_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def main_legacy(data_file: str):
    """
    Legacy mode: Load from JSONL file.

    Args:
        data_file: Path to JSONL export file
    """
    print("="*80)
    print("BASELINE EVALUATION - Random Shuffle (Legacy Mode)")
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

    # Split data chronologically using legacy method
    from chronological_splitter import split_multiple_users

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

    return all_results


def main_database(weeks: int = 4, min_interactions_per_week: int = 5):
    """
    Database mode: Load from PostgreSQL database.

    Args:
        weeks: Number of weeks to look back for active users
        min_interactions_per_week: Minimum interactions per week threshold
    """
    print("="*80)
    print("BASELINE EVALUATION - Random Shuffle (Database Mode)")
    print("="*80)
    print(f"\nActive user criteria: ≥{min_interactions_per_week} interactions/week for {weeks} weeks\n")

    # Initialize data fetcher
    fetcher = UserDataFetcher()

    # Get data summary
    summary = fetcher.get_data_summary()
    print(f"Database Summary:")
    print(f"  Total records: {summary['total_records']}")
    print(f"  Unique users: {summary['unique_users']}")
    print(f"  Date range: {summary['date_range'][0]} to {summary['date_range'][1]}\n")

    # Fetch active user timelines
    timelines = fetcher.fetch_active_user_timelines(
        weeks=weeks,
        min_interactions_per_week=min_interactions_per_week
    )

    if not timelines:
        print("No active users found. Try expanding time window or lowering threshold.")
        return {}

    # Split data chronologically (train/val/test 70/15/15)
    print("Splitting data chronologically (70/15/15 train/val/test)...\n")
    user_splits = split_user_timelines_three_way(timelines, train_ratio=0.7, val_ratio=0.15)

    print(f"Successfully split data for {len(user_splits)} users\n")

    # Filter to only users with positive feedbacks in test set
    users_with_positives = []
    for user_name, split in user_splits.items():
        # Count positive feedbacks in test set
        test_positives = sum(len(item.get_positive_papers()) for item in split.test)
        if test_positives > 0:
            users_with_positives.append(user_name)

    print(f"Users with positive feedbacks in test set: {users_with_positives}")
    print(f"Number of users to evaluate: {len(users_with_positives)}\n")

    # Evaluate baseline for each user
    # Merge all test queries into one per user
    all_results = {}

    for user_name, split in user_splits.items():
        # Skip users without positive feedbacks in test set
        if user_name not in users_with_positives:
            print(f"\nSkipping {user_name} (no positive feedbacks in test set)")
            continue

        print(f"\n{'='*80}")
        print(f"Evaluating: {user_name}")
        print(f"{'='*80}")
        print_split_summary(split, user_name)

        print(f"Test set size: {len(split.test)} queries (will be merged into 1)")

        # Merge all test queries into a single candidate pool
        merged_candidates = []
        merged_labels = {}

        for item in split.test:
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
            timestamp=split.test[-1].timestamp,  # Use latest timestamp
            user_name=user_name,
            query_context=f"Merged test set for {user_name}",
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

    return all_results


def main():
    """Main entry point - auto-detect mode based on environment"""
    # Check if database credentials are available
    db_host = os.getenv("DB_HOST")
    legacy_file = "/Users/bran/Desktop/AIgnite-Solutions/AIgnite/experiments/data/result_1214/export_user_retrieve_results.jsonl"

    if db_host:
        # Database mode
        logger.info("Database credentials detected, using database mode")
        all_results = main_database(weeks=4, min_interactions_per_week=5)
    elif Path(legacy_file).exists():
        # Legacy mode
        logger.info(f"No database credentials, using legacy file mode: {legacy_file}")
        all_results = main_legacy(legacy_file)

        # Overall statistics across all users (legacy)
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
    else:
        logger.error("Neither database credentials nor legacy JSONL file found")
        logger.info("Set DB_* environment variables or provide JSONL file")
        return


if __name__ == "__main__":
    main()
