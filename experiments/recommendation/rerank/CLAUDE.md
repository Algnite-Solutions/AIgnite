# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the **rerank experiment** folder for AIgnite's paper recommendation system. It focuses on LLM-based reranking of academic papers using user feedback data. The system uses PDF first pages as input for both profile extraction and reranking.

## Environment Setup

Copy `.env.template` to `.env` and configure:

```bash
cp .env.template .env
```

Required environment variables:
- `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME` - PostgreSQL credentials
- `GEMINI_API_KEY` - Google Gemini API key for LLM operations
- `ARXIV_PDF_CACHE_DIR` - Directory for PDF caching (default: `./pdfs_cache`)
- `ARXIV_RATE_LIMIT_DELAY` - Delay between ArXiv API calls in seconds (default: 3.0)

## Pipeline Workflow

The standard workflow has 3 steps:

### Step 1: Prepare User Data (Cache PDFs)
```bash
source .env
python prepare_user_data.py --user "Qi Zhu"
```
- Downloads PDFs for user's positive feedback days
- Saves metadata to `results/{user}_metadata.json`

### Step 2: Generate Profile
```bash
source .env
python generate_profile.py --user "Qi Zhu"
```
- Reads metadata + cached PDFs
- Extracts user profile using `Verbalizer` (PDF-based)
- Saves profile to `results/profile_{user}.json`
- Outputs token usage

### Step 3: Evaluate Profile
```bash
source .env
python evaluate_profile.py --user "Qi Zhu" --print
```
- Loads profile + PDFs
- Reranks using `GeminiRerankerPDF` with personalized prompt
- Calculates F1 on validation days

## Main Scripts

**`prepare_user_data.py`**: Download and cache PDFs
- Fetches positive days from `paper_recommendations` table
- Collects candidates from `user_retrieve_results`
- Downloads PDFs with rate limiting
- Output: `results/{user}_metadata.json`

**`generate_profile.py`**: Extract user profile
- Reads metadata, builds training data (70/30 split)
- Uses `Verbalizer.extract_profile()` with PDF first pages
- Output: `results/profile_{user}.json` with token usage

**`evaluate_profile.py`**: Evaluate profile performance
- Loads profile, reranks validation candidates
- Uses `personalized_ranking_prompt` from `rerank_prompts.yaml`
- Calculates Precision/Recall/F1

**`compare_ranking.py`**: Compare reranking vs original top_k
- For January 2026 days
- Reranks `retrieve_ids` (~20) and compares top 5 vs original `top_k_ids`
- Outputs daily differences and metrics

## Source Modules (`src/AIgnite/recommendation/`)

**`Verbalizer.py`**:
- `extract_profile(training_data, pdf_paths_dict, max_papers)` → (profile, usage)
- Uses PDF first pages as multimodal input
- Returns profile dict + token usage

**`LLMReranker.py`**:
- `GeminiRerankerPDF` - PDF-based reranking
- `rerank(prompt_value_dict, pdf_paths_dict, retrieve_ids, top_k)`
- `prompt_value_dict` contains values for template placeholders

**`rerank_prompts.yaml`**:
- `profile_extraction_prompt` - Profile extraction
- `personalized_ranking_prompt` - Reranking with profile
- `personalized_subset_selection_prompt` - Subset selection

## Database Schema

**`user_retrieve_results`**:
- `retrieve_ids` - All retrieved paper IDs (~20)
- `top_k_ids` - Top-k shown to user (5)

**`paper_recommendations`**:
- `blog_liked` - Positive feedback marker

## Code Organization

- `scratchpad/` - POC code, experimental scripts, one-off analysis
- `results/` - Experiment outputs (profiles, metrics, metadata)
- `pdfs_cache/` - Cached PDF files

## Important Notes

1. **PDF Input**: Both Verbalizer and GeminiRerankerPDF use PDF first pages as multimodal input (not text extraction)

2. **Import Path**:
   ```python
   sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
   from AIgnite.recommendation import GeminiRerankerPDF, Verbalizer
   ```

3. **Token Tracking**: `generate_profile.py` outputs token usage for cost monitoring

4. **Rate Limiting**: Use `ArxivPDFFetcher` for PDF downloads to respect ArXiv API limits
