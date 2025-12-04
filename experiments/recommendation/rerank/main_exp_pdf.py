import sys
import os
import json
from pathlib import Path

# Add the src directory to the path so we can import from AIgnite
from AIgnite.recommendation import GeminiRerankerPDF

if __name__ == "__main__":

    data_path = "/Users/bran/Desktop/AIgnite-Solutions/AIgnite/experiments/data/user_feedbacks_1128/"
    pdf_dir = os.path.join(data_path, "pdfs")

    # Build mapping of document IDs to PDF paths
    pdf_paths_dict = {}
    query_set = []

    print("Scanning for PDF files...")
    print("Note: Only the first page of each PDF will be sent to Gemini for reranking.")
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")

    for pdf_path in pdf_files:
        # Extract paper ID from filename (e.g., "2506.14948v1.pdf" -> "2506.14948v1")
        corpus_id = pdf_path.stem
        pdf_paths_dict[corpus_id] = str(pdf_path)

    print(f"Mapped {len(pdf_paths_dict)} PDF files")

    print("Loading queries...")
    with open(f"{data_path}export_user_retrieve_results.jsonl") as IN:
        for line in IN:
            tmp = json.loads(line)
            query_set.append({"query": tmp['query'], "retrieve_ids": tmp['retrieved_ids']})
    print(f"Loaded {len(query_set)} queries")

    # Initialize reranker
    print("Initializing PDF reranker...")
    reranker = GeminiRerankerPDF()

    # Process queries and rerank
    results = []
    for i, item in enumerate(query_set[::-1][:5]):
        print(f"\n{'='*80}")
        print(f"Processing query {i+1}/{len(query_set)}")
        print(f"Query: {item['query'][:150]}...")
        print(f"{'='*80}")

        reranked_ids, thought_summary = reranker.rerank(
            query=item['query'],
            pdf_paths_dict=pdf_paths_dict,
            retrieve_ids=item['retrieve_ids'][:20],
            top_k=5
        )

        # Compare with original top_k_ids
        original_top_k = item.get('top_k_ids', item['retrieve_ids'][:5])

        print(f"\n📊 Original Top-5:")
        for j, doc_id in enumerate(original_top_k[:5], 1):
            pdf_status = "✓ PDF available" if doc_id in pdf_paths_dict else "✗ PDF missing"
            print(f"  {j}. {doc_id} [{pdf_status}]")

        print(f"\n🎯 Reranked Top-5:")
        for j, doc_id in enumerate(reranked_ids[:5], 1):
            marker = "✓ KEPT" if doc_id in original_top_k[:5] else "⭐ NEW"
            pdf_status = "✓ PDF available" if doc_id in pdf_paths_dict else "✗ PDF missing"
            print(f"  {j}. {doc_id} [{marker}] [{pdf_status}]")

        # Show differences
        original_set = set(original_top_k[:5])
        reranked_set = set(reranked_ids[:5])
        removed = original_set - reranked_set
        added = reranked_set - original_set

        if removed:
            print(f"\n❌ Removed from top-5: {', '.join(removed)}")
        if added:
            print(f"\n✨ Added to top-5: {', '.join(added)}")

        results.append({
            "query": item['query'],
            "original_retrieve_ids": item['retrieve_ids'],
            "original_top_k_ids": original_top_k,
            "reranked_ids": reranked_ids,
            "thought_summary": thought_summary
        })

        if len(results) >= 5:
            break


    # Save results
    output_file = "rerank_results_pdf.jsonl"
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w') as OUT:
        for result in results:
            OUT.write(json.dumps(result) + '\n')

    print("Done!")
