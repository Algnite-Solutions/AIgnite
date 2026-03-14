import sys
import unittest
import os
from unittest.mock import MagicMock, patch
from pathlib import Path

# Adjust path to import AIgnite modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

sys.modules['PyPDF2'] = MagicMock()
sys.modules['google'] = MagicMock()
sys.modules['google.genai'] = MagicMock()
sys.modules['google.genai.types'] = MagicMock()

from AIgnite.recommendation.LLMReranker import GeminiRerankerPDF  # noqa: E402

class TestGeminiRerankerPDF(unittest.TestCase):
    @patch("AIgnite.recommendation.LLMReranker.genai.Client")
    def test_rerank_with_user_profile(self, mock_client_class):
        # Mock genai Client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "<Documents>\nsub_id_1\n</Documents>"
        # For thought summary if enable_thinking=True
        mock_candidate = MagicMock()
        mock_part = MagicMock()
        mock_part.thought = True
        mock_part.text = "Thinking about documents."
        mock_candidate.content.parts = [mock_part]
        mock_response.candidates = [mock_candidate]
        mock_client.models.generate_content.return_value = mock_response

        # Instantiate reranker
        os.environ["GEMINI_API_KEY"] = "fake_key"
        reranker = GeminiRerankerPDF(model_name="test-model", prompt_key="blog_rerank_pdf_prompt")

        # Define fake inputs
        query = "Test user query"
        pdf_paths_dict = {}
        retrieve_ids = ["doc_1", "doc_2"]
        user_profile = {
            "persona_definition": "Test Researcher",
            "negative_constraints": ["Constraint 1", "Constraint 2"],
            "ranking_heuristics": ["Heuristic A", "Heuristic B"]
        }

        with patch("AIgnite.recommendation.LLMReranker.extract_first_page_pdf") as mock_extract:
            mock_extract.return_value = b"fake pdf content"
            # Call rerank
            ranked, thoughts = reranker.rerank(
                query=query, 
                pdf_paths_dict=pdf_paths_dict, 
                retrieve_ids=retrieve_ids,
                user_profile=user_profile
            )

            # Check that model was called
            mock_client.models.generate_content.assert_called_once()
            
            # Verify contents formatting
            call_args = mock_client.models.generate_content.call_args
            contents = call_args.kwargs.get('contents', [])
            
            # The last item in contents should be the prompt string
            prompt_str = contents[-1]
            self.assertIn("Test Researcher", prompt_str)
            self.assertIn("- Constraint 1", prompt_str)
            self.assertIn("- Constraint 2", prompt_str)
            self.assertIn("- Heuristic A", prompt_str)
            self.assertIn("- Heuristic B", prompt_str)
            self.assertIn("Test user query", prompt_str)
            self.assertEqual(ranked, ["sub_id_1"])

    @patch("AIgnite.recommendation.LLMReranker.genai.Client")
    def test_rerank_without_user_profile(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "<Documents>\ndoc_2\n</Documents>"
        mock_response.candidates = []
        mock_client.models.generate_content.return_value = mock_response

        os.environ["GEMINI_API_KEY"] = "fake_key"
        reranker = GeminiRerankerPDF(model_name="test-model", prompt_key="blog_rerank_pdf_prompt")

        query = "Test basic query"
        pdf_paths_dict = {}
        retrieve_ids = ["doc_1", "doc_2"]

        with patch("AIgnite.recommendation.LLMReranker.extract_first_page_pdf"):  # noqa: F841
            ranked, thoughts = reranker.rerank(
                query=query, 
                pdf_paths_dict=pdf_paths_dict, 
                retrieve_ids=retrieve_ids
            )
            
            call_args = mock_client.models.generate_content.call_args
            contents = call_args.kwargs.get('contents', [])
            prompt_str = contents[-1]
            
            self.assertNotIn("Persona Definition:", prompt_str)
            self.assertIn("Test basic query", prompt_str)
            self.assertEqual(ranked, ["doc_2"])

if __name__ == "__main__":
    unittest.main()