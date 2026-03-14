"""
Verbalizer module for extracting user preferences from feedback data.
Uses PDF first pages as input (consistent with LLMReranker).
"""

import os
import json
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
import PyPDF2
import io

logger = logging.getLogger(__name__)


def extract_first_page_pdf(pdf_path: str) -> Optional[bytes]:
    """
    Extract the first page from a PDF file and return it as bytes.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Bytes of a new PDF containing only the first page, or None if extraction fails
    """
    if PyPDF2 is None:
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


class Verbalizer:
    """Extract user preferences from feedback examples using PDF content."""

    def __init__(self, model_name="gemini-2.5-flash"):
        """
        Initialize the Verbalizer with a Gemini model.

        Args:
            model_name: Name of the Gemini model to use (default: gemini-2.5-flash)
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

        # Load prompts from YAML
        prompt_path = Path(__file__).parent / "rerank_prompts.yaml"
        with open(prompt_path, 'r') as f:
            self.prompts = yaml.safe_load(f)

        logger.info(f"Initialized Verbalizer with model: {model_name}")

    def extract_profile(
        self,
        training_data: List[Dict],
        pdf_paths_dict: Dict[str, str],
        max_papers: int = 10
    ) -> tuple[Dict[str, Any], Dict[str, int]]:
        """
        Extract user profile from training data with PDF first pages.

        Args:
            training_data: List of training days, each with 'day', 'query', 'candidates'
                          where candidates have 'paper_id' and 'label' (1=liked, 0=not)
            pdf_paths_dict: Dict mapping paper_id to PDF file path
            max_papers: Maximum number of papers to include (default: 10)

        Returns:
            Tuple of (profile, usage) where:
            - profile: Dict with 'persona_definition', 'negative_constraints', 'ranking_heuristics'
            - usage: Dict with 'input_tokens', 'thoughts_token_count', 'total_tokens'
        """
        # Build contents with PDF parts
        contents = self._build_pdf_contents(
            training_data, pdf_paths_dict, max_papers=max_papers
        )

        # Add the text prompt
        prompt = self.prompts['profile_extraction_prompt'].format(
            training_examples="[PDF papers are provided above with their IDs and labels]"
        )
        contents.append(prompt)

        logger.info(f"Extracting profile from {len(training_data)} training days")

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents
            )
            profile = self._parse_json_response(response.text)

            # Extract token usage
            usage = {
                'input_tokens': 0,
                'thoughts_tokens': 0,
                'total_tokens': 0
            }
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage['input_tokens'] = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
                usage['thoughts_tokens'] = getattr(response.usage_metadata, 'thoughts_token_count', 0) or 0
                usage['total_tokens'] = getattr(response.usage_metadata, 'total_token_count', 0) or 0

            logger.info(f"Successfully extracted profile (tokens: {usage['total_tokens']})")
            return profile, usage
        except Exception as e:
            logger.error(f"Failed to extract profile: {e}")
            return self._get_default_profile(), {'input_tokens': 0, 'thoughts_tokens': 0, 'total_tokens': 0}

    def _build_pdf_contents(
        self,
        training_data: List[Dict],
        pdf_paths_dict: Dict[str, str],
        max_papers: int = 10
    ) -> List:
        """
        Build contents list with PDF parts for Gemini API.

        Args:
            training_data: List of training days with candidates
            pdf_paths_dict: Dict mapping paper_id to PDF file path
            max_papers: Maximum papers to include

        Returns:
            List of content parts (PDF bytes + text labels)
        """
        contents = []
        paper_count = 0

        for day_data in training_data:
            if paper_count >= max_papers:
                break

            day_str = day_data.get('day', 'Unknown')
            query = day_data.get('query', 'Unknown')

            contents.append(f"\n=== Day: {day_str} ===")
            contents.append(f"Query: {query}\n")

            # Separate positives and negatives
            positives = [c for c in day_data.get('candidates', []) if c.get('label') == 1]
            negatives = [c for c in day_data.get('candidates', []) if c.get('label') == 0]

            # Add positive papers with PDFs
            if positives:
                contents.append("--- LIKED Papers (User gave positive feedback) ---")
                for pos in positives[:3]:  # Limit to 3 positives per day
                    if paper_count >= max_papers:
                        break
                    paper_id = pos.get('paper_id')
                    if paper_id in pdf_paths_dict:
                        first_page = extract_first_page_pdf(pdf_paths_dict[paper_id])
                        if first_page:
                            contents.append(f"\n[Paper ID: {paper_id}] - LIKED")
                            contents.append(
                                types.Part.from_bytes(
                                    data=first_page,
                                    mime_type='application/pdf'
                                )
                            )
                            paper_count += 1

            # Add negative papers (limit to 2 per day)
            if negatives and paper_count < max_papers:
                contents.append("\n--- SHOWN but NOT LIKED Papers ---")
                for neg in negatives:
                    if paper_count >= max_papers:
                        break
                    paper_id = neg.get('paper_id')
                    if paper_id in pdf_paths_dict:
                        first_page = extract_first_page_pdf(pdf_paths_dict[paper_id])
                        if first_page:
                            contents.append(f"\n[Paper ID: {paper_id}] - NOT LIKED")
                            contents.append(
                                types.Part.from_bytes(
                                    data=first_page,
                                    mime_type='application/pdf'
                                )
                            )
                            paper_count += 1

        logger.info(f"Built contents with {paper_count} PDF papers")
        return contents

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response.

        Args:
            response_text: Raw response text from LLM

        Returns:
            Parsed JSON dict

        Raises:
            json.JSONDecodeError: If response is not valid JSON
        """
        # Try to extract JSON from markdown code blocks
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
        else:
            json_str = response_text.strip()

        return json.loads(json_str)

    def _get_default_profile(self) -> Dict[str, Any]:
        """Return default profile when extraction fails."""
        return {
            "persona_definition": "Researcher interested in machine learning and AI",
            "negative_constraints": ["Avoid irrelevant topics"],
            "ranking_heuristics": ["Prioritize recent papers", "Prefer empirical results"],
            "confidence": "low"
        }
