"""
PDF Reranking Experiment - Minimal version focused on k=5.
"""

import json
import os
from pathlib import Path
from collections import defaultdict

from AIgnite.recommendation import GeminiRerankerPDF
from data_schema import UserTimeline, FeedbackItem
from chronological_splitter import split_multiple_users
from evaluation_metrics import evaluate_ranking


def main():
    # Setup paths
    data_file = "/Users/bran/Desktop/AIgnite-Solutions/AIgnite/experiments/data/result_1214/export_user_retrieve_results.jsonl"
    pdf_dir = "/Users/bran/Desktop/AIgnite-Solutions/AIgnite/experiments/data/result_1214/pdfs"

    # Build PDF mapping
    pdf_paths_dict = {}
    for pdf_path in Path(pdf_dir).glob("*.pdf"):
        pdf_paths_dict[pdf_path.stem] = str(pdf_path)
    print(f"Loaded {len(pdf_paths_dict)} PDFs\n")

    # Load and split user data
    user_entries = defaultdict(list)
    with open(data_file) as f:
        for line in f:
            entry = json.loads(line)
            user_entries[entry['user_name']].append(entry)

    user_timelines = [UserTimeline.from_jsonl_entries(entries)
                      for entries in user_entries.values()]
    user_splits = split_multiple_users(user_timelines, train_ratio=0.8)

    # Filter to users with positive feedbacks
    users_to_test = [t.user_name for t in user_timelines
                     if len(t.get_all_positive_papers()) > 0 and t.user_name in user_splits]
    print(f"Testing {len(users_to_test)} users: {users_to_test}\n")

    # Initialize reranker
    reranker = GeminiRerankerPDF()

    # Process each user
    results = []
    for user_name in users_to_test:
        split = user_splits[user_name]

        # Merge dev queries
        merged_candidates = []
        merged_labels = {}
        for item in split.dev:
            merged_candidates.extend(item.candidate_set)
            merged_labels.update(item.labels)

        # Remove duplicates
        unique_candidates = []
        for c in merged_candidates:
            if c not in unique_candidates:
                unique_candidates.append(c)

        # Filter to candidates with PDFs
        candidates_with_pdfs = [c for c in unique_candidates if c in pdf_paths_dict]

        print(f"{user_name}:")
        print(f"  Candidates: {len(unique_candidates)}, With PDFs: {len(candidates_with_pdfs)}")
        print(f"  Positives: {sum(1 for k,v in merged_labels.items() if v==1)}")

        if len(candidates_with_pdfs) < 5:
            print(f"  Skipping - not enough PDFs\n")
            continue

        # Rerank
        if False:
            latest_query = split.dev[-1].query_context
            reranked_ids, _ = reranker.rerank(
                query=latest_query,
                pdf_paths_dict=pdf_paths_dict,
                retrieve_ids=candidates_with_pdfs,  # Use top 20 as input
                top_k=5
            )
        breakpoint()
        # Evaluate at k=5
        baseline_metrics = evaluate_ranking(unique_candidates,
            FeedbackItem(timestamp=split.dev[-1].timestamp, user_name=user_name,
                        query_context=latest_query, candidate_set=unique_candidates,
                        labels=merged_labels), k_values=[5])

        reranked_metrics = evaluate_ranking(reranked_ids + unique_candidates[5:],
            FeedbackItem(timestamp=split.dev[-1].timestamp, user_name=user_name,
                        query_context=latest_query, candidate_set=unique_candidates,
                        labels=merged_labels), k_values=[5])

        print(f"  Baseline  - Recall@5: {baseline_metrics['recall@5']:.3f}, NDCG@5: {baseline_metrics['ndcg@5']:.3f}")
        print(f"  Reranked  - Recall@5: {reranked_metrics['recall@5']:.3f}, NDCG@5: {reranked_metrics['ndcg@5']:.3f}")
        print(f"  Top 5: {reranked_ids}\n")

        results.append({
            'user': user_name,
            'baseline': baseline_metrics,
            'reranked': reranked_metrics,
            'top_5': reranked_ids
        })

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    avg_baseline_recall = sum(r['baseline']['recall@5'] for r in results) / len(results)
    avg_baseline_ndcg = sum(r['baseline']['ndcg@5'] for r in results) / len(results)
    avg_reranked_recall = sum(r['reranked']['recall@5'] for r in results) / len(results)
    avg_reranked_ndcg = sum(r['reranked']['ndcg@5'] for r in results) / len(results)

    print(f"Baseline  - Recall@5: {avg_baseline_recall:.3f}, NDCG@5: {avg_baseline_ndcg:.3f}")
    print(f"Reranked  - Recall@5: {avg_reranked_recall:.3f}, NDCG@5: {avg_reranked_ndcg:.3f}")
    print(f"Improvement - Recall@5: {avg_reranked_recall-avg_baseline_recall:+.3f}, "
          f"NDCG@5: {avg_reranked_ndcg-avg_baseline_ndcg:+.3f}")


if __name__ == "__main__":
    main()
