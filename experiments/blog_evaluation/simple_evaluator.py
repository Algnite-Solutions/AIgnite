#!/usr/bin/env python3
"""
Simple Blog Evaluator using the simplified prompt template.

This evaluator uses a streamlined approach to compare two blog posts
against their source scientific paper across three dimensions:
helpfulness, comprehensiveness, and factual grounding.
"""

import json
import os
import glob
import argparse
import yaml
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Set, Tuple
from pathlib import Path

# Import the generator from the main codebase
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
from AIgnite.generation.generator import GeminiGenerator

CRITERIA = '''```json
{{"helpfulness": {{
    "winner": "A" | "B" | "TIE",
    "confidence": [1-5],
    "margin": "slight" | "significant",
    "reasoning": "Detailed explanation of why one blog is more helpful than the other",
    "specific_differences": [
      "Specific difference that makes the winner better",
      "Another key difference"
    ]}},
  "comprehensiveness": {{"winner": "A" | "B" | "TIE",
    "confidence": [1-5],
    "margin": "slight" | "significant",
    "reasoning": "Detailed explanation of why one blog is more comprehensive than the other",
    "specific_differences": [
      "Specific difference that makes the winner better",
      "Another key difference"
    ]}},
  "factual_grounding": {{"winner": "A" | "B" | "TIE",
    "confidence": [1-5],
    "margin": "slight" | "significant",
    "reasoning": "Detailed explanation of why one blog is more factually grounded than the other",
    "specific_differences": [
      "Specific difference that makes the winner better",
      "Another key difference"
    ]}}
}}
```'''


@dataclass
class SimpleEvaluation:
    """Result of evaluating one dimension of blog comparison"""
    winner: str  # "A", "B", or "TIE"
    confidence: int  # 1-5 scale
    margin: str  # "slight" or "significant"
    reasoning: str
    specific_differences: List[str]


@dataclass
class SimpleComparisonResult:
    """Complete comparison result using simplified evaluation"""
    helpfulness: SimpleEvaluation
    comprehensiveness: SimpleEvaluation
    factual_grounding: SimpleEvaluation
    paper_id: Optional[str] = None


class SimpleBlogEvaluator:
    """Simple blog evaluator using the streamlined prompt approach"""
    
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Initialize the simple evaluator.
        
        Args:
            model_name: Name of the model to use for evaluation
        """
        self.generator = GeminiGenerator(model_name=model_name)
        self.prompt_template = self._load_prompt_template()
    
    def _load_prompt_template(self) -> str:
        """Load the simplified prompt template"""
        prompt_path = Path(__file__).parent / "resources" / "simplified_prompt.txt"
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def evaluate(self, paper_source: str, blog_a_text: str, blog_b_text: str, 
                paper_title: str = "") -> SimpleComparisonResult:
        """
        Evaluate two blogs against their source paper using simplified approach.
        
        Args:
            paper_source: PDF path, later support docSet
            blog_a_text: Text content of blog A
            blog_b_text: Text content of blog B
            paper_title: Optional title of the paper
            
        Returns:
            SimpleComparisonResult containing evaluation across all dimensions
        """
        # Read paper content if it's a file path
        if os.path.isfile(paper_source):
            pass
        else:
            raise ValueError("PDF should be read")

        # Prepare the prompt
        prompt = self.prompt_template.format(blog_a_text=blog_a_text, blog_b_text=blog_b_text) + CRITERIA

        # Generate evaluation
        try:
            
            response = self.generator.generate_response(prompt, pdf_path = paper_source)
            evaluation_data = self._parse_json_response(response)
            
            # Convert to dataclass
            result = SimpleComparisonResult(
                helpfulness=SimpleEvaluation(**evaluation_data["helpfulness"]),
                comprehensiveness=SimpleEvaluation(**evaluation_data["comprehensiveness"]),
                factual_grounding=SimpleEvaluation(**evaluation_data["factual_grounding"]),
                paper_id=paper_title
            )
            
            return result
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            # Return fallback result
            return self._create_fallback_result(paper_title, str(e))
    
    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON response from the model"""
        try:
            # Try to extract JSON from the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[start_idx:end_idx]
            return json.loads(json_str)
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Response was: {response}")
            raise
    
    def _create_fallback_result(self, paper_id: str, error_msg: str) -> SimpleComparisonResult:
        """Create a fallback result when evaluation fails"""
        fallback_eval = SimpleEvaluation(
            winner="TIE",
            confidence=1,
            margin="slight",
            reasoning=f"Evaluation failed: {error_msg}",
            specific_differences=["Unable to complete evaluation due to error"]
        )
        
        return SimpleComparisonResult(
            helpfulness=fallback_eval,
            comprehensiveness=fallback_eval,
            factual_grounding=fallback_eval,
            paper_id=paper_id
        )
    
    def generate_report(self, result: SimpleComparisonResult) -> str:
        """Generate a readable report from the evaluation result"""
        
        def format_evaluation(name: str, eval_result: SimpleEvaluation) -> str:
            winner_text = f"Blog {eval_result.winner}" if eval_result.winner != "TIE" else "TIE"
            return f"""### {name}
**Winner:** {winner_text} (Confidence: {eval_result.confidence}/5, Margin: {eval_result.margin})

**Reasoning:** {eval_result.reasoning}

**Key Differences:**
{chr(10).join(f"• {diff}" for diff in eval_result.specific_differences)}"""
        
        report = f"""# Simple Blog Evaluation Report
{f"**Paper:** {result.paper_id}" if result.paper_id else ""}

## Evaluation Results

{format_evaluation("Helpfulness", result.helpfulness)}

{format_evaluation("Comprehensiveness", result.comprehensiveness)}

{format_evaluation("Factual Grounding", result.factual_grounding)}

## Summary
- **Helpfulness Winner:** Blog {result.helpfulness.winner if result.helpfulness.winner != "TIE" else "TIE"}
- **Comprehensiveness Winner:** Blog {result.comprehensiveness.winner if result.comprehensiveness.winner != "TIE" else "TIE"}  
- **Factual Grounding Winner:** Blog {result.factual_grounding.winner if result.factual_grounding.winner != "TIE" else "TIE"}

---
*Generated by Simple Blog Evaluator*
"""
        return report


def parse_file_paths(path_a: str, path_b: str, path_pdf: str) -> List[Tuple[str, str, str, str]]:
    """
    Parse files from three directories and find matching paper IDs.
    
    Args:
        path_a: Directory containing blogs from model A (.md files)
        path_b: Directory containing blogs from model B (.md files) 
        path_pdf: Directory containing PDF files (.pdf files)
    
    Returns:
        List of tuples (paper_id, blog_a_path, blog_b_path, pdf_path) for matching files
    """
    # Get all paper IDs from each directory
    model_a_files = {}
    model_b_files = {}
    pdf_files = {}
    
    # Parse model A blog files
    for file_path in glob.glob(os.path.join(path_a, "*.md")):
        paper_id = os.path.splitext(os.path.basename(file_path))[0]
        model_a_files[paper_id] = file_path
    
    # Parse model B blog files  
    for file_path in glob.glob(os.path.join(path_b, "*.md")):
        paper_id = os.path.splitext(os.path.basename(file_path))[0]
        model_b_files[paper_id] = file_path
    
    # Parse PDF files
    for file_path in glob.glob(os.path.join(path_pdf, "*.pdf")):
        paper_id = os.path.splitext(os.path.basename(file_path))[0]
        pdf_files[paper_id] = file_path
    
    # Find paper IDs that exist in all three directories
    all_paper_ids = set(model_a_files.keys()).intersection(
        set(model_b_files.keys())
    ).intersection(set(pdf_files.keys()))
    
    # Create list of matching file paths
    matching_files = []
    for paper_id in sorted(all_paper_ids):
        matching_files.append((
            paper_id,
            model_a_files[paper_id],
            model_b_files[paper_id], 
            pdf_files[paper_id]
        ))
    
    print(f"Found {len(model_a_files)} Model A blog files")
    print(f"Found {len(model_b_files)} Model B blog files")
    print(f"Found {len(pdf_files)} PDF files")
    print(f"Found {len(matching_files)} matching paper sets")
    
    return matching_files


def batch_evaluate_from_paths(path_a: str, path_b: str, path_pdf: str, 
                             output_dir: Optional[str] = None,
                             model_name: str = "gemini-2.5-flash") -> List[SimpleComparisonResult]:
    """
    Batch evaluate blogs from three directories.
    
    Args:
        path_a: Directory containing blogs from model A
        path_b: Directory containing blogs from model B
        path_pdf: Directory containing PDF files
        output_dir: Optional directory to save individual reports
        model_name: Model name for the evaluator
        
    Returns:
        List of evaluation results
    """
    # Parse file paths
    matching_files = parse_file_paths(path_a, path_b, path_pdf)
    
    if not matching_files:
        print("No matching files found across all three directories!")
        return []
    
    # Initialize evaluator
    evaluator = SimpleBlogEvaluator(model_name=model_name)
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    results = []
    successful_evaluations = 0
    failed_evaluations = 0
    
    print(f"\nStarting batch evaluation of {len(matching_files)} papers...")
    print("=" * 60)
    
    for i, (paper_id, blog_a_path, blog_b_path, pdf_path) in enumerate(matching_files, 1):
        print(f"\n[{i}/{len(matching_files)}] Processing: {paper_id}")
        print("-" * 40)
        
        try:

            # Read blog contents
            with open(blog_a_path, 'r', encoding='utf-8') as f:
                blog_a_text = f.read()
                
            with open(blog_b_path, 'r', encoding='utf-8') as f:
                blog_b_text = f.read()
            
            # Run evaluation
            result = evaluator.evaluate(
                paper_source=pdf_path,
                blog_a_text=blog_a_text,
                blog_b_text=blog_b_text,
                paper_title=paper_id
            )
            
            results.append(result)
            successful_evaluations += 1
            
            # Save individual report if output directory specified
            if output_dir:
                report = evaluator.generate_report(result)
                output_file = os.path.join(output_dir, f"{paper_id}_evaluation.md")
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report)
                
                print(f"Report saved to: {output_file}")
            
        except Exception as e:
            print(f"ERROR evaluating {paper_id}: {str(e)}")
            failed_evaluations += 1
            continue
    
    print("\n" + "=" * 60)
    print(f"Batch evaluation completed!")
    print(f"Successful evaluations: {successful_evaluations}")
    print(f"Failed evaluations: {failed_evaluations}")
    print(f"Total papers processed: {len(matching_files)}")
    
    return results


def parse_arguments():
    """Parse command line arguments for the simple evaluator"""
    parser = argparse.ArgumentParser(
        description="Simple blog evaluator using streamlined prompt approach"
    )
    
    parser.add_argument(
        "--path-a",
        required=True,
        help="Directory containing blogs from model A (.md files)"
    )
    
    parser.add_argument(
        "--path-b", 
        required=True,
        help="Directory containing blogs from model B (.md files)"
    )
    
    parser.add_argument(
        "--path-pdf",
        required=True,
        help="Directory containing PDF files (.pdf files)"
    )
    
    parser.add_argument(
        "--output-dir",
        help="Directory to save evaluation reports"
    )
    
    parser.add_argument(
        "--model-name",
        default="gemini-2.5-flash",
        help="Model name for evaluation (default: gemini-2.5-flash)"
    )
    
    parser.add_argument(
        "--list-papers",
        action="store_true",
        help="List available matching paper IDs and exit"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for command-line usage"""
    args = parse_arguments()
    
    print(f"Model A blogs path: {args.path_a}")
    print(f"Model B blogs path: {args.path_b}")
    print(f"PDFs path: {args.path_pdf}")
    print("-" * 50)
    
    # Validate directory paths exist
    for path_name, path in [("Path A", args.path_a), ("Path B", args.path_b), ("Path C", args.path_pdf)]:
        if not os.path.isdir(path):
            print(f"Error: {path_name} directory does not exist: {path}")
            sys.exit(1)
    
    # List papers mode
    if args.list_papers:
        matching_files = parse_file_paths(args.path_a, args.path_b, args.path_pdf)
        if matching_files:
            print(f"\nAvailable paper IDs ({len(matching_files)}):")
            for paper_id, _, _, _ in matching_files:
                print(f"  - {paper_id}")
        else:
            print("\nNo matching paper IDs found!")
        return
    
    # Run batch evaluation
    try:
        results = batch_evaluate_from_paths(
            path_a=args.path_a,
            path_b=args.path_b,
            path_pdf=args.path_pdf,
            output_dir=args.output_dir,
            model_name=args.model_name
        )
        
        print(f"\nBatch evaluation completed! Processed {len(results)} papers.")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()