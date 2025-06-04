import unittest
from unittest.mock import MagicMock, patch
from AIgnite.generation.generator import GeminiBlogGenerator
from AIgnite.data.docset import DocSet, TextChunk

class TestGeminiBlogGenerator(unittest.TestCase):
    def setUp(self):
        self.sample_paper = DocSet(
            doc_id="paper_001",
            title="Sample Paper",
            abstract="Abstract here.",
            authors=["Alice", "Bob"],
            categories=["cs.AI"],
            published_date="2024-01-01",
            pdf_path="test/data/2501.11216.pdf",  # provide a small dummy PDF file
            text_chunks=[TextChunk(id="t1", type="text", text="This is a test chunk.")],
            figure_chunks=[],
            table_chunks=[],
            metadata={}
        )

    @patch("AIgnite.generation.generator.genai.Client")
    def test_generate_digest(self, MockGenAIClient):
        mock_client_instance = MockGenAIClient.return_value
        mock_response = MagicMock()
        mock_response.text = "# Sample Blog Post\nThis is a test."
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 300

        mock_client_instance.models.generate_content.return_value = mock_response

        generator = GeminiBlogGenerator(output_path="./")
        generator.client = mock_client_instance

        generator.generate_digest([self.sample_paper])

        mock_client_instance.models.generate_content.assert_called_once()
        print("✅ generate_digest ran successfully")

if __name__ == "__main__":
    unittest.main()