"""
LLM-based reranker module for academic document retrieval.
Provides both text-based and PDF-based reranking using Google Gemini models.
"""

import os
import re
import yaml
from pathlib import Path
from google import genai
from google.genai import types

import PyPDF2
import io


def extract_first_page_pdf(pdf_path):
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
        print(f"Warning: Failed to extract first page from {pdf_path}: {e}")
        return None


class GeminiReranker:
    """
    A class to rerank retrieved documents using the Gemini model with text input.
    This class uses the Google Gemini model to rerank academic papers based on relevance to a query.
    """
    def __init__(self, model_name="gemini-2.5-pro", prompt_key="blog_rerank_prompt"):
        """
        Initialize the Gemini reranker.

        Args:
            model_name: Name of the Gemini model to use
            prompt_key: Key in rerank_prompts.yaml to use for the prompt template
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

        # Load prompts
        prompt_path = Path(__file__).parent / "rerank_prompts.yaml"
        with open(prompt_path, 'r') as f:
            prompts = yaml.safe_load(f)
            self.rerank_prompt_template = prompts[prompt_key]

    def rerank(self, query, corpus_dict, retrieve_ids, top_k=5):
        """
        Rerank retrieved documents based on query relevance.

        Args:
            query: The user's research question
            corpus_dict: Dictionary mapping document IDs to document content (text)
            retrieve_ids: List of retrieved document IDs to rerank
            top_k: Number of top documents to return (default: 5)

        Returns:
            List of top_k reranked document IDs
        """
        # Build documents text
        documents_text = ""
        for doc_id in retrieve_ids:
            if doc_id in corpus_dict:
                documents_text += f"Document ID: {doc_id}\n{corpus_dict[doc_id]}\n\n"

        # Format prompt
        prompt = self.rerank_prompt_template.format(
            documents_text=documents_text,
            query=query
        )

        # Call Gemini API
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )

        # Parse response to extract document IDs
        response_text = response.text
        doc_ids = self._parse_document_ids(response_text)

        return doc_ids[:top_k]

    def _parse_document_ids(self, response_text):
        """
        Parse document IDs from the model's response.
        Expects format: <Documents>doc_id_1\ndoc_id_2\n...</Documents>
        """
        # Extract content between <Documents> tags
        match = re.search(r'<Documents>(.*?)</Documents>', response_text, re.DOTALL)
        if match:
            doc_ids_text = match.group(1).strip()
            doc_ids = [line.strip() for line in doc_ids_text.split('\n') if line.strip()]
            return doc_ids
        return []


class GeminiRerankerPDF:
    """
    A class to rerank retrieved documents using the Gemini model with PDF support.
    This version passes only the first page of PDF files directly to Gemini's API.
    """
    def __init__(self, model_name="gemini-2.5-pro", prompt_key="blog_rerank_pdf_prompt",
                 enable_thinking=True):
        """
        Initialize the PDF-based Gemini reranker.

        Args:
            model_name: Name of the Gemini model to use
            prompt_key: Key in rerank_prompts.yaml to use for the prompt template
            enable_thinking: Whether to enable thinking mode (captures reasoning process)
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.enable_thinking = enable_thinking

        # Load prompts
        prompt_path = Path(__file__).parent / "rerank_prompts.yaml"
        with open(prompt_path, 'r') as f:
            prompts = yaml.safe_load(f)
            self.rerank_prompt_template = prompts[prompt_key]

    def rerank(self, query, pdf_paths_dict, retrieve_ids, top_k=5):
        """
        Rerank retrieved documents based on query relevance using PDF first pages.

        Args:
            query: The user's research question
            pdf_paths_dict: Dictionary mapping document IDs to PDF file paths
            retrieve_ids: List of retrieved document IDs to rerank
            top_k: Number of top documents to return (default: 5)

        Returns:
            Tuple of (reranked_doc_ids, thought_summary)
            - reranked_doc_ids: List of top_k reranked document IDs
            - thought_summary: String containing AI thinking process (empty if not enabled)
        """
        # Build the content list with PDFs and document IDs
        contents = []

        # Build documents_text section with PDF parts
        documents_text_parts = []

        for doc_id in retrieve_ids:
            if doc_id in pdf_paths_dict:
                pdf_path = pdf_paths_dict[doc_id]

                # Extract first page from PDF
                try:
                    first_page_pdf = extract_first_page_pdf(pdf_path)

                    if first_page_pdf is None:
                        print(f"Warning: Could not extract first page from {pdf_path}")
                        continue

                    # Add document ID label before each PDF
                    documents_text_parts.append(f"\n=== Document ID: {doc_id} ===\n")

                    # Add first page PDF as a Part
                    documents_text_parts.append(
                        types.Part.from_bytes(
                            data=first_page_pdf,
                            mime_type='application/pdf',
                        )
                    )
                except Exception as e:
                    print(f"Warning: Failed to process PDF {pdf_path}: {e}")
                    continue

        # Now construct the prompt using the template
        # First add all the document parts
        contents.extend(documents_text_parts)

        # Then add the formatted prompt
        prompt = self.rerank_prompt_template.format(
            documents_text="[PDFs are provided above with Document IDs]",
            user_interest_description=query
        )

        contents.append(prompt)

        # Call Gemini API
        try:
            config_params = {}
            if self.enable_thinking:
                config_params['thinking_config'] = types.ThinkingConfig(
                    include_thoughts=True
                )

            response = self.client.models.generate_content(
                model=self.model_name,
                config=types.GenerateContentConfig(**config_params) if config_params else None,
                contents=contents
            )

            # Extract thinking summary if available
            thought_summary = ""
            if self.enable_thinking:
                if hasattr(response, 'candidates') and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'thought') and part.thought:
                                thought_summary = part.text
                                print(f"\n=� Thought summary captured ({len(thought_summary)} chars)")
                                break

            # Parse response to extract document IDs
            response_text = response.text
            doc_ids = self._parse_document_ids(response_text)

            return doc_ids[:top_k], thought_summary
        except Exception as e:
            print(f"Error during reranking: {e}")
            return retrieve_ids[:top_k], ""  # Return original order as fallback

    def _parse_document_ids(self, response_text):
        """
        Parse document IDs from the model's response.
        Expects format: <Documents>doc_id_1\ndoc_id_2\n...</Documents>
        """
        # Extract content between <Documents> tags
        match = re.search(r'<Documents>(.*?)</Documents>', response_text, re.DOTALL)
        if match:
            doc_ids_text = match.group(1).strip()
            doc_ids = [line.strip() for line in doc_ids_text.split('\n') if line.strip()]
            return doc_ids
        return []
