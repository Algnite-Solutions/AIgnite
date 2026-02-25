"""
Evaluate user profile on validation data.

Reads profile from generate_profile.py output and cached PDFs to evaluate
paper recommendations using GeminiRerankerPDF with personalized prompts.

Usage:
    source .env

    # Evaluate profile with default settings
    python evaluate_profile.py --user "Qi Zhu"

    # Specify profile file
    python evaluate_profile.py --profile profile_Qi_Zhu.json

    # Use different prompt template
    python evaluate_profile.py --user "Qi Zhu" --prompt personalized_ranking_prompt

    # Print detailed per-day results
    python evaluate_profile.py --user "Qi Zhu" --print
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Set

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from AIgnite.recommendation.LLMReranker import GeminiRerankerPDF
from pdf_fetcher import ArxivPDFFetcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_profile(profile_path: Path) -> Dict:
    """Load profile from generate_profile.py output."""
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile file not found: {profile_path}")

    with open(profile_path, 'r') as f:
        data = json.load(f)

    logger.info(f"Loaded profile from {profile_path}")
    return data


def load_metadata(metadata_path: Path) -> Dict:
    """Load metadata from prepare_user_data.py output."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    logger.info(f"Loaded metadata from {metadata_path}")
    return metadata


def build_val_data(
    metadata: Dict,
    val_days: List[str]
) -> List[Dict]:
    """
    Build validation data for evaluation.

    Args:
        metadata: Metadata dict from prepare_user_data.py
        val_days: List of validation day strings

    Returns:
        List of validation examples, each with 'day', 'candidates', 'positive_ids'
    """
    val_data = []
    positive_days = metadata.get('positive_days', {})

    for day in val_days:
        day_data = positive_days.get(day, {})
        positives = day_data.get('positive_papers', [])
        candidates = day_data.get('candidates', [])

        positive_ids = [p['paper_id'] for p in positives]

        val_data.append({
            'day': day,
            'candidates': candidates,
            'positive_ids': positive_ids
        })

    logger.info(f"Built {len(val_data)} validation examples")
    return val_data


def build_pdf_paths_dict(
    paper_ids: List[str],
    cache_dir: Path,
    pdf_fetcher: ArxivPDFFetcher
) -> Dict[str, str]:
    """Build dict mapping paper_id to PDF file path."""
    pdf_paths = {}

    for paper_id in paper_ids:
        normalized_id = pdf_fetcher._normalize_arxiv_id(paper_id)
        cache_path = cache_dir / f"{normalized_id}.pdf"

        if cache_path.exists():
            pdf_paths[paper_id] = str(cache_path)

    return pdf_paths


def calculate_f1(predicted: Set[str], actual: Set[str]) -> Dict:
    """Calculate Precision, Recall, F1."""
    if not predicted and not actual:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}

    if not predicted:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    if not actual:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    true_positives = len(predicted & actual)
    precision = true_positives / len(predicted)
    recall = true_positives / len(actual)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1}


def evaluate_on_val(
    val_data: List[Dict],
    profile: Dict,
    cache_dir: Path,
    prompt_key: str = 'personalized_subset_selection_prompt',
    print_details: bool = False
) -> Dict:
    """
    Evaluate profile on validation days using F1 metric.

    Args:
        val_data: List of validation examples
        profile: User profile dict
        cache_dir: PDF cache directory
        prompt_key: Key in rerank_prompts.yaml to use
        print_details: Whether to print per-day details

    Returns:
        Dict with avg_precision, avg_recall, avg_f1, per_day_results
    """
    pdf_fetcher = ArxivPDFFetcher(cache_dir=str(cache_dir))
    reranker = GeminiRerankerPDF(
        model_name="gemini-3.1-pro-preview",
        prompt_key=prompt_key,
        enable_thinking=True
    )

    results = []

    for day_item in val_data:
        day_str = day_item['day']
        candidates = day_item['candidates']
        actual_positives = set(day_item['positive_ids'])

        if not actual_positives:
            continue

        # Build PDF paths for candidates
        pdf_paths = build_pdf_paths_dict(candidates, cache_dir, pdf_fetcher)

        if len(pdf_paths) < len(candidates):
            logger.warning(f"Day {day_str}: Only {len(pdf_paths)}/{len(candidates)} PDFs available")

        if not pdf_paths:
            continue

        prompt_value_dict = {
            'persona_definition': profile.get('persona_definition', 'N/A'),
            'negative_constraints': "\n".join(f"- {c}" for c in profile.get('negative_constraints', [])) or "None",
            'ranking_heuristics': "\n".join(f"- {h}" for h in profile.get('ranking_heuristics', [])) or "None",
        }
        try:
            # Call reranker
            ranked, _ = reranker.rerank(
                query=prompt_value_dict,
                pdf_paths_dict=pdf_paths,
                retrieve_ids=list(pdf_paths.keys()),
                top_k=len(pdf_paths)
            )

            # Treat all returned as "selected"
            predicted = set(ranked) if ranked else set()

        except Exception as e:
            logger.error(f"Error evaluating day {day_str}: {e}")
            predicted = set()

        # Calculate metrics
        metrics = calculate_f1(predicted, actual_positives)

        result = {
            'day': day_str,
            'candidates': candidates,
            'predicted': list(predicted),
            'actual': list(actual_positives),
            **metrics
        }
        results.append(result)

        if print_details:
            logger.info(f"\nDay {day_str}:")
            logger.info(f"  Candidates: {len(candidates)}")
            logger.info(f"  Actual positives: {list(actual_positives)}")
            logger.info(f"  Predicted: {list(predicted)}")
            logger.info(f"  P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1']:.2f}")

    # Aggregate metrics
    if results:
        avg_precision = sum(r['precision'] for r in results) / len(results)
        avg_recall = sum(r['recall'] for r in results) / len(results)
        avg_f1 = sum(r['f1'] for r in results) / len(results)
    else:
        avg_precision = avg_recall = avg_f1 = 0.0

    return {
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
        'num_days': len(results),
        'per_day_results': results
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate user profile on validation data")
    parser.add_argument('--user', type=str, default='Qi Zhu', help='User name')
    parser.add_argument('--profile', type=str, default=None,
                        help='Path to profile JSON (default: profile_{user}.json)')
    parser.add_argument('--metadata', type=str, default=None,
                        help='Path to metadata JSON (default: results/{user}_metadata.json)')
    parser.add_argument('--cache-dir', type=str, default='./pdfs_cache',
                        help='PDF cache directory')
    parser.add_argument('--prompt', type=str, default='personalized_subset_selection_prompt',
                        help='Prompt template key to use')
    parser.add_argument('--output', type=str, default=None,
                        help='Output results JSON path')
    parser.add_argument('--print', action='store_true',
                        help='Print detailed per-day results')

    args = parser.parse_args()

    # Resolve paths
    cache_dir = Path(args.cache_dir)
    profile_path = Path(args.profile) if args.profile else \
                   Path(f"results/profile_{args.user.replace(' ', '_')}.json")
    print("Reading Profile", profile_path)
    metadata_path = Path(args.metadata) if args.metadata else \
                    Path(f"results/{args.user.replace(' ', '_')}_metadata.json")
    output_path = Path(args.output) if args.output else \
                  Path(f"results/results_{args.user.replace(' ', '_')}.json")

    # Load profile
    try:
        profile_data = load_profile(profile_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("Run generate_profile.py first to create profile")
        return 1

    profile = profile_data.get('profile', profile_data)
    val_days = profile_data.get('val_days', [])

    if not val_days:
        # Try to get val_days from metadata
        try:
            metadata = load_metadata(metadata_path)
            positive_days = sorted(metadata.get('positive_days', {}).keys())
            train_ratio = profile_data.get('train_ratio', 0.7)
            split_idx = int(len(positive_days) * train_ratio)
            val_days = positive_days[split_idx:]
        except FileNotFoundError:
            logger.error("No val_days in profile and metadata not found")
            return 1

    # Load metadata for validation data
    try:
        metadata = load_metadata(metadata_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    # Build validation data
    val_data = build_val_data(metadata, val_days)

    if not val_data:
        logger.error("No validation data found")
        return 1

    logger.info(f"\nEvaluating on {len(val_data)} validation days...")

    # Evaluate
    results = evaluate_on_val(
        val_data=val_data,
        profile=profile,
        cache_dir=cache_dir,
        prompt_key=args.prompt,
        print_details=args.print
    )

    # Print summary
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    print(f"\nUser: {args.user}")
    print(f"Prompt: {args.prompt}")
    print(f"Val Days: {results['num_days']}")
    print(f"\nPrecision: {results['avg_precision']:.3f}")
    print(f"Recall: {results['avg_recall']:.3f}")
    print(f"F1: {results['avg_f1']:.3f}")
    print("="*60 + "\n")

    # Save results
    output = {
        'user': args.user,
        'profile_file': str(profile_path),
        'prompt_used': args.prompt,
        'val_days': val_days,
        'results': {
            'precision': results['avg_precision'],
            'recall': results['avg_recall'],
            'f1': results['avg_f1'],
            'num_days': results['num_days'],
            'per_day': results['per_day_results']
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
