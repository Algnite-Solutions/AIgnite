"""
Generate user profile from training data.

Reads metadata from prepare_user_data.py and cached PDFs to extract
user preferences using Verbalizer.

Usage:
    source .env

    # Generate profile with default 70/30 train/val split
    python generate_profile.py --user "Qi Zhu"

    # Custom train ratio
    python generate_profile.py --user "Qi Zhu" --train-ratio 0.8

    # Specify metadata file
    python generate_profile.py --user "Qi Zhu" --metadata results/Qi_Zhu_metadata.json

    # Custom max papers for profile extraction
    python generate_profile.py --user "Qi Zhu" --max-papers 15
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from AIgnite.recommendation.Verbalizer import Verbalizer
from pdf_fetcher import ArxivPDFFetcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_metadata(metadata_path: Path) -> Dict:
    """Load metadata from prepare_user_data.py output."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    logger.info(f"Loaded metadata from {metadata_path}")
    logger.info(f"  Positive days: {metadata.get('positive_days_count', 0)}")
    logger.info(f"  Total candidates: {metadata.get('total_candidates', 0)}")

    return metadata


def split_days_chronologically(
    positive_days: List[str],
    train_ratio: float = 0.7
) -> Tuple[List[str], List[str]]:
    """
    Split days chronologically into train/val sets.

    Args:
        positive_days: Sorted list of day strings (YYYY-MM-DD)
        train_ratio: Proportion for training (default: 0.7)

    Returns:
        Tuple of (train_days, val_days)
    """
    split_idx = int(len(positive_days) * train_ratio)
    train_days = positive_days[:split_idx]
    val_days = positive_days[split_idx:]

    logger.info(f"Split: {len(train_days)} train days, {len(val_days)} val days")

    return train_days, val_days


def build_training_data(
    metadata: Dict,
    train_days: List[str]
) -> List[Dict]:
    """
    Build training data for Verbalizer.

    Args:
        metadata: Metadata dict from prepare_user_data.py
        train_days: List of training day strings

    Returns:
        List of training examples, each with 'day', 'query', 'candidates'
    """
    logger.info("Building training data...")

    training_data = []
    positive_days = metadata.get('positive_days', {})

    for day in train_days:
        day_data = positive_days.get(day, {})
        positives = day_data.get('positive_papers', [])
        candidates = day_data.get('candidates', [])

        positive_ids = set(p['paper_id'] for p in positives)

        labeled = []
        for paper_id in candidates:
            labeled.append({
                'paper_id': paper_id,
                'label': 1 if paper_id in positive_ids else 0
            })

        # Use first positive's query if available, else generic
        query = positives[0].get('title', f"Papers from {day}") if positives else f"Papers from {day}"

        training_data.append({
            'day': day,
            'query': query,
            'candidates': labeled
        })

    logger.info(f"Built {len(training_data)} training examples")
    return training_data


def build_pdf_paths_dict(
    paper_ids: List[str],
    cache_dir: Path,
    pdf_fetcher: ArxivPDFFetcher
) -> Dict[str, str]:
    """
    Build dict mapping paper_id to PDF file path.

    Args:
        paper_ids: List of paper IDs
        cache_dir: Directory containing cached PDFs
        pdf_fetcher: ArxivPDFFetcher instance

    Returns:
        Dict mapping paper_id to PDF path
    """
    pdf_paths = {}

    for paper_id in paper_ids:
        normalized_id = pdf_fetcher._normalize_arxiv_id(paper_id)
        cache_path = cache_dir / f"{normalized_id}.pdf"

        if cache_path.exists():
            pdf_paths[paper_id] = str(cache_path)

    logger.info(f"Found {len(pdf_paths)}/{len(paper_ids)} cached PDFs")
    return pdf_paths


def main():
    parser = argparse.ArgumentParser(description="Generate user profile from training data")
    parser.add_argument('--user', type=str, default='Qi Zhu', help='User name')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Train split ratio (default 0.7)')
    parser.add_argument('--metadata', type=str, default=None,
                        help='Path to metadata JSON (default: results/{user}_metadata.json)')
    parser.add_argument('--cache-dir', type=str, default='./pdfs_cache',
                        help='PDF cache directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output profile JSON path (default: results/profile_{user}.json)')
    parser.add_argument('--max-papers', type=int, default=10,
                        help='Max papers to include in profile extraction (default: 10)')
    parser.add_argument('--model', type=str, default='gemini-3.1-pro-preview',
                        help='Gemini model to use')

    args = parser.parse_args()

    # Resolve paths
    cache_dir = Path(args.cache_dir)
    metadata_path = Path(args.metadata) if args.metadata else \
                    Path(f"results/{args.user.replace(' ', '_')}_metadata.json")
    output_path = Path(args.output) if args.output else \
                  Path(f"results/profile_{args.user.replace(' ', '_')}.json")

    # Load metadata
    try:
        metadata = load_metadata(metadata_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("Run prepare_user_data.py first to generate metadata")
        return 1

    # Get sorted positive days
    positive_days = sorted(metadata.get('positive_days', {}).keys())

    if len(positive_days) < 2:
        logger.error(f"Not enough positive days: {len(positive_days)}, need at least 2")
        return 1

    # Split train/val
    train_days, val_days = split_days_chronologically(positive_days, args.train_ratio)

    # Build training data
    training_data = build_training_data(metadata, train_days)

    # Collect all paper IDs from training data
    all_paper_ids = set()
    for item in training_data:
        for c in item.get('candidates', []):
            all_paper_ids.add(c['paper_id'])

    # Build PDF paths dict
    pdf_fetcher = ArxivPDFFetcher(cache_dir=str(cache_dir))
    pdf_paths_dict = build_pdf_paths_dict(list(all_paper_ids), cache_dir, pdf_fetcher)

    if len(pdf_paths_dict) < 3:
        logger.error(f"Not enough cached PDFs: {len(pdf_paths_dict)}, need at least 3")
        logger.error("Run prepare_user_data.py to download PDFs")
        return 1

    # Extract profile using Verbalizer
    logger.info(f"\nExtracting profile using {args.model}...")
    verbalizer = Verbalizer(model_name=args.model)

    profile, usage = verbalizer.extract_profile(
        training_data=training_data,
        pdf_paths_dict=pdf_paths_dict,
        max_papers=args.max_papers
    )

    # Build output
    output = {
        'user': args.user,
        'train_ratio': args.train_ratio,
        'train_days': train_days,
        'val_days': val_days,
        'profile': profile,
        'token_usage': usage,
        'generated_at': datetime.now().isoformat(),
        'config': {
            'model': args.model,
            'max_papers': args.max_papers,
            'metadata_file': str(metadata_path)
        }
    }

    # Save profile
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nProfile saved to {output_path}")

    # Print profile summary
    print("\n" + "="*60)
    print("GENERATED PROFILE")
    print("="*60)
    print(f"\nUser: {args.user}")
    print(f"Model: {args.model}")
    print(f"Train days: {len(train_days)}, Val days: {len(val_days)}")
    print(f"\nToken Usage: {usage['total_tokens']:,} total ({usage['input_tokens']:,} input + {usage['thoughts_tokens']:,} output)")
    print(f"\nPersona: {profile.get('persona_definition', 'N/A')}")
    print("\nNegative Constraints:")
    for c in profile.get('negative_constraints', []):
        print(f"  - {c}")
    print("\nRanking Heuristics:")
    for h in profile.get('ranking_heuristics', []):
        print(f"  - {h}")
    print(f"\nConfidence: {profile.get('confidence', 'N/A')}")
    print("="*60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
