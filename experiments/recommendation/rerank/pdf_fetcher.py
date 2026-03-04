"""
ArXiv PDF fetching with local caching.

This module provides on-demand PDF downloading from ArXiv API with
local caching and rate limiting to avoid API abuse.
"""

import os
import time
import hashlib
import logging
import io
from typing import List, Optional, Dict, Set
from pathlib import Path

import arxiv

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

logger = logging.getLogger(__name__)


class ArxivPDFFetcher:
    """
    Fetch ArXiv PDFs on-demand with local caching and rate limiting.

    Features:
        - Local file caching to avoid re-downloading
        - Rate limiting (3 second delay between requests)
        - Track missing PDFs to avoid repeated failures
        - Cache statistics
    """

    def __init__(
        self,
        cache_dir: str = "./pdfs_cache",
        rate_limit_delay: float = 3.0
    ):
        """
        Initialize PDF fetcher.

        Args:
            cache_dir: Directory to cache downloaded PDFs (default: ./pdfs_cache)
            rate_limit_delay: Delay between API calls in seconds (default: 3.0)
        """
        self.cache_dir = Path(cache_dir)
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0.0
        self.missing_pdfs: Set[str] = set()  # Track PDFs that don't exist

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized ArxivPDFFetcher with cache_dir={self.cache_dir}")

    def fetch_pdf(self, arxiv_id: str) -> Optional[str]:
        """
        Fetch PDF for a given ArXiv ID.

        Args:
            arxiv_id: ArXiv paper ID (e.g., "2301.12345" or "cs.AI/1234")

        Returns:
            Local path to PDF file, or None if unavailable
        """
        # Normalize ArXiv ID
        arxiv_id = self._normalize_arxiv_id(arxiv_id)

        # Check if we know this PDF doesn't exist
        if arxiv_id in self.missing_pdfs:
            logger.debug(f"Skipping {arxiv_id} (previously failed)")
            return None

        # Check local cache first
        cached_path = self._get_cache_path(arxiv_id)
        if cached_path.exists():
            logger.debug(f"Cache hit: {arxiv_id}")
            return str(cached_path)

        # Rate limiting
        self._rate_limit()

        # Try to download
        try:
            logger.info(f"Downloading PDF for {arxiv_id}...")

            # Search for paper by ID
            client = arxiv.Client()
            search = arxiv.Search(id_list=[arxiv_id])
            result = next(client.results(search), None)

            if not result:
                logger.warning(f"Paper {arxiv_id} not found on ArXiv")
                self.missing_pdfs.add(arxiv_id)
                return None

            # Download PDF
            result.download_pdf(dirpath=str(self.cache_dir), filename=f"{arxiv_id}.pdf")

            downloaded_path = self.cache_dir / f"{arxiv_id}.pdf"

            if downloaded_path.exists():
                logger.info(f"✅ Downloaded: {arxiv_id} -> {downloaded_path}")
                return str(downloaded_path)
            else:
                logger.warning(f"Download failed for {arxiv_id}")
                self.missing_pdfs.add(arxiv_id)
                return None

        except Exception as e:
            logger.error(f"Failed to download PDF for {arxiv_id}: {e}")
            self.missing_pdfs.add(arxiv_id)
            return None

    def fetch_multiple_pdfs(self, arxiv_ids: List[str]) -> Dict[str, Optional[str]]:
        """
        Fetch multiple PDFs with rate limiting.

        Args:
            arxiv_ids: List of ArXiv paper IDs

        Returns:
            Dictionary mapping arxiv_id to local path (or None if unavailable)
        """
        results = {}

        for arxiv_id in arxiv_ids:
            path = self.fetch_pdf(arxiv_id)
            results[arxiv_id] = path

        return results

    def get_cache_stats(self) -> Dict[str, any]:
        """
        Get statistics about the PDF cache.

        Returns:
            Dictionary with cached_files, total_size_mb, missing_pdfs
        """
        cached_files = list(self.cache_dir.glob("*.pdf"))
        total_size = sum(f.stat().st_size for f in cached_files)

        return {
            'cached_files': len(cached_files),
            'total_size_mb': total_size / (1024 * 1024),
            'missing_pdfs': len(self.missing_pdfs),
            'cache_dir': str(self.cache_dir)
        }

    def clear_cache(self) -> int:
        """
        Clear all cached PDFs.

        Returns:
            Number of files deleted
        """
        cached_files = list(self.cache_dir.glob("*.pdf"))
        count = 0

        for f in cached_files:
            try:
                f.unlink()
                count += 1
            except Exception as e:
                logger.error(f"Failed to delete {f}: {e}")

        logger.info(f"Cleared {count} cached PDFs")
        return count

    def _normalize_arxiv_id(self, arxiv_id: str) -> str:
        """
        Normalize ArXiv ID format.

        Args:
            arxiv_id: Raw ArXiv ID (may have URL components, etc.)

        Returns:
            Normalized ArXiv ID (e.g., "2301.12345")
        """
        # Remove URL components
        arxiv_id = arxiv_id.strip()

        # Remove common prefixes
        for prefix in ["http://arxiv.org/abs/", "https://arxiv.org/abs/",
                       "http://arxiv.org/pdf/", "https://arxiv.org/pdf/",
                       "arxiv:", "arXiv:"]:
            if arxiv_id.startswith(prefix):
                arxiv_id = arxiv_id[len(prefix):]

        # Remove .pdf suffix if present
        if arxiv_id.endswith(".pdf"):
            arxiv_id = arxiv_id[:-4]

        return arxiv_id

    def _get_cache_path(self, arxiv_id: str) -> Path:
        """Get local cache path for an ArXiv ID"""
        normalized_id = self._normalize_arxiv_id(arxiv_id)
        return self.cache_dir / f"{normalized_id}.pdf"

    def _rate_limit(self):
        """Enforce rate limiting between API calls"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            logger.debug(f"Rate limit: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def extract_first_page_text(self, pdf_path: str) -> Optional[str]:
        """
        Extract text content from the first page of a PDF.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text from first page, or None if extraction fails
        """
        if not HAS_PYPDF2:
            logger.warning("PyPDF2 not available, cannot extract text")
            return None

        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                if len(pdf_reader.pages) == 0:
                    return None

                # Extract text from first page
                first_page = pdf_reader.pages[0]
                text = first_page.extract_text()

                # Clean up text
                text = text.replace('\n', ' ').replace('\r', ' ')
                text = ' '.join(text.split())  # Normalize whitespace

                return text

        except Exception as e:
            logger.warning(f"Failed to extract text from {pdf_path}: {e}")
            return None

    def extract_first_page_pdf(self, pdf_path: str) -> Optional[bytes]:
        """
        Extract the first page of a PDF as a new PDF (bytes).

        Useful for sending to LLMs that accept PDF uploads.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Bytes of a new PDF containing only the first page, or None if extraction fails
        """
        if not HAS_PYPDF2:
            logger.warning("PyPDF2 not available, cannot extract PDF page")
            return None

        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                if len(pdf_reader.pages) == 0:
                    return None

                # Create a new PDF with only the first page
                pdf_writer = PyPDF2.PdfWriter()
                pdf_writer.add_page(pdf_reader.pages[0])

                # Write to bytes
                output_buffer = io.BytesIO()
                pdf_writer.write(output_buffer)
                output_buffer.seek(0)

                return output_buffer.read()

        except Exception as e:
            logger.warning(f"Failed to extract first page from {pdf_path}: {e}")
            return None

    def fetch_paper_metadata(self, arxiv_id: str) -> Optional[Dict]:
        """
        Fetch paper metadata from ArXiv.

        Args:
            arxiv_id: ArXiv paper ID

        Returns:
            Dict with 'title', 'authors', 'summary', 'url', or None if not found
        """
        arxiv_id = self._normalize_arxiv_id(arxiv_id)

        try:
            self._rate_limit()

            client = arxiv.Client()
            search = arxiv.Search(id_list=[arxiv_id])
            result = next(client.results(search), None)

            if result:
                return {
                    'arxiv_id': arxiv_id,
                    'title': result.title,
                    'authors': [str(a) for a in result.authors],
                    'summary': result.summary,
                    'url': result.entry_id,
                    'published': str(result.published),
                    'categories': result.categories
                }
            return None

        except Exception as e:
            logger.error(f"Failed to fetch metadata for {arxiv_id}: {e}")
            return None
