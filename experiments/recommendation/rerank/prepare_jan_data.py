"""
Prepare January 2026 data for annotation/comparison.

Fetches PDFs for all papers in retrieve_ids for January 2026 days.

Usage:
    source .env
    python prepare_jan_data.py --user "Qi Zhu"
    python prepare_jan_data.py --user "Qi Zhu" --dry-run
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from sqlalchemy import func
from sqlalchemy.orm import Session

from db_config import DatabaseConfig
from data_fetcher import UserRetrieveResult, PaperRecommendation
from pdf_fetcher import ArxivPDFFetcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_jan_2026_days(session: Session, user_name: str) -> list[dict]:
    """
    Get all January 2026 days with retrieve_ids and top_k_ids.

    Returns:
        List of dicts with 'day', 'query', 'retrieve_ids', 'top_k_ids'
    """
    # Query for January 2026
    records = session.query(UserRetrieveResult)\
        .filter(UserRetrieveResult.username == user_name)\
        .filter(func.extract('year', UserRetrieveResult.recommendation_date) == 2026)\
        .filter(func.extract('month', UserRetrieveResult.recommendation_date) == 1)\
        .order_by(UserRetrieveResult.recommendation_date)\
        .all()

    # Group by date
    days_data = defaultdict(lambda: {'queries': [], 'retrieve_ids': set(), 'top_k_ids': set()})

    for record in records:
        day = record.recommendation_date.date()
        days_data[day]['queries'].append(record.query)
        if record.retrieve_ids:
            days_data[day]['retrieve_ids'].update(record.retrieve_ids)
        if record.top_k_ids:
            days_data[day]['top_k_ids'].update(record.top_k_ids)

    # Convert to list
    result = []
    for day, data in sorted(days_data.items()):
        result.append({
            'day': str(day),
            'queries': data['queries'],
            'retrieve_ids': list(data['retrieve_ids']),
            'top_k_ids': list(data['top_k_ids'])
        })

    logger.info(f"Found {len(result)} days in January 2026")
    return result


def get_positive_ids_for_days(session: Session, user_name: str, days: list[str]) -> dict[str, list[str]]:
    """Get positive feedback (blog_liked) for each day."""
    positives_by_day = {}

    for day_str in days:
        day_date = datetime.strptime(day_str, '%Y-%m-%d').date()

        recs = session.query(PaperRecommendation)\
            .filter(PaperRecommendation.username == user_name)\
            .filter(func.date(PaperRecommendation.recommendation_date) == day_date)\
            .filter(PaperRecommendation.blog_liked == True)\
            .all()

        positives_by_day[day_str] = [r.paper_id for r in recs]

    return positives_by_day


def cache_pdfs_for_days(
    days_data: list[dict],
    cache_dir: Path,
    dry_run: bool = False
) -> dict[str, str]:
    """Download and cache PDFs for all papers in days_data."""
    pdf_fetcher = ArxivPDFFetcher(cache_dir=str(cache_dir))

    # Collect all unique paper IDs
    all_paper_ids = set()
    for day_item in days_data:
        all_paper_ids.update(day_item['retrieve_ids'])
        all_paper_ids.update(day_item['top_k_ids'])

    logger.info(f"Total unique papers to cache: {len(all_paper_ids)}")

    cache_info = {}
    successful = 0
    failed = 0
    already_cached = 0

    for i, paper_id in enumerate(sorted(all_paper_ids), 1):
        normalized_id = pdf_fetcher._normalize_arxiv_id(paper_id)
        cache_path = cache_dir / f"{normalized_id}.pdf"

        if cache_path.exists():
            cache_info[paper_id] = {'status': 'cached', 'path': str(cache_path)}
            already_cached += 1
            continue

        if dry_run:
            logger.info(f"[{i}/{len(all_paper_ids)}] Would download: {paper_id}")
            cache_info[paper_id] = {'status': 'would_download', 'path': None}
            continue

        logger.info(f"[{i}/{len(all_paper_ids)}] Downloading: {paper_id}")
        try:
            pdf_path = pdf_fetcher.fetch_pdf(paper_id)
            if pdf_path:
                cache_info[paper_id] = {'status': 'downloaded', 'path': pdf_path}
                successful += 1
            else:
                cache_info[paper_id] = {'status': 'failed', 'path': None}
                failed += 1
        except Exception as e:
            logger.warning(f"Failed to fetch {paper_id}: {e}")
            cache_info[paper_id] = {'status': 'failed', 'path': None}
            failed += 1

    logger.info(f"\nPDF Caching Summary:")
    logger.info(f"  Total papers: {len(all_paper_ids)}")
    logger.info(f"  Already cached: {already_cached}")
    logger.info(f"  Newly downloaded: {successful}")
    logger.info(f"  Failed: {failed}")

    return cache_info


def main():
    parser = argparse.ArgumentParser(description="Prepare January 2026 data")
    parser.add_argument('--user', type=str, default='Qi Zhu', help='User name')
    parser.add_argument('--cache-dir', type=str, default='./pdfs_cache',
                        help='PDF cache directory')
    parser.add_argument('--output', type=str, default='./jan_2026_data.json',
                        help='Output JSON path')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be downloaded without downloading')

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    output_path = Path(args.output)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Connect to database
    db_config = DatabaseConfig()
    session = db_config.create_session()

    try:
        # Get January 2026 days
        days_data = get_jan_2026_days(session, args.user)

        if not days_data:
            logger.error("No January 2026 data found")
            return 1

        # Get positive feedback for those days
        day_strs = [d['day'] for d in days_data]
        positives_by_day = get_positive_ids_for_days(session, args.user, day_strs)

        # Add positives to days_data
        for day_item in days_data:
            day_item['positive_ids'] = positives_by_day.get(day_item['day'], [])

        # Cache PDFs
        cache_info = cache_pdfs_for_days(days_data, cache_dir, dry_run=args.dry_run)

        # Save output
        output = {
            'user': args.user,
            'month': '2026-01',
            'preparation_date': datetime.now().isoformat(),
            'days_count': len(days_data),
            'days': days_data,
            'cache_stats': {
                'total': len(cache_info),
                'cached': sum(1 for v in cache_info.values() if v['status'] == 'cached'),
                'downloaded': sum(1 for v in cache_info.values() if v['status'] == 'downloaded'),
                'failed': sum(1 for v in cache_info.values() if v['status'] == 'failed'),
            },
            'cache_info': cache_info
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"\nData saved to {output_path}")
        print(f"\nSummary: {len(days_data)} days, {cache_info.__len__()} papers")

        return 0

    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        return 1
    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
