import sys
import os
import json


from AIgnite.recommendation import GeminiReranker


if __name__ == "__main__":
    data_path = "../../data/user_feedbacks_1128/"
    # Load corpus
    corpus_dict = {}
    query_set = []

    print("Loading corpus...")
    with open(f"{data_path}arxiv_metadata.tsv") as IN:
        for line in IN:
            corpus_id, content = line.strip().split("\t")
            corpus_dict[corpus_id] = content
    print(f"Loaded {len(corpus_dict)} documents")

    print("Loading queries...")
    with open(f"{data_path}export_user_retrieve_results.jsonl") as IN:
        for line in IN:
            tmp = json.loads(line)
            query_set.append({"query": tmp['query'], "retrieve_ids": tmp['retrieved_ids']})
    print(f"Loaded {len(query_set)} queries")

    # Initialize reranker
    print("Initializing reranker...")
    reranker = GeminiReranker()

    # Process queries and rerank
    results = []
    for i, item in enumerate(query_set):
        print(f"\n{'='*80}")
        print(f"Processing query {i+1}/{len(query_set)}")
        print(f"Query: {item['query'][:150]}...")
        print(f"{'='*80}")
        #breakpoint()
        reranked_ids = reranker.rerank(
            query=item['query'],
            corpus_dict=corpus_dict,
            retrieve_ids=item['retrieve_ids'][:20],
            top_k=5
        )

        # Compare with original top_k_ids
        original_top_k = item.get('top_k_ids', item['retrieve_ids'][:5])

        print(f"\n📊 Original Top-5:")
        for j, doc_id in enumerate(original_top_k[:5], 1):
            content_preview = corpus_dict.get(doc_id, "N/A")[:200]
            print(f"  {j}. {doc_id}")
            print(f"     {content_preview}...")

        print(f"\n🎯 Reranked Top-5:")
        for j, doc_id in enumerate(reranked_ids[:5], 1):
            content_preview = corpus_dict.get(doc_id, "N/A")[:200]
            marker = "✓ KEPT" if doc_id in original_top_k[:5] else "⭐ NEW"
            print(f"  {j}. {doc_id} [{marker}]")
            print(f"     {content_preview}...")

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
            "reranked_ids": reranked_ids
        })
        if len(results) > 5:
            break


    # Save results
    output_file = "rerank_results.jsonl"
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w') as OUT:
        for result in results:
            OUT.write(json.dumps(result) + '\n')

    print("Done!")
    
