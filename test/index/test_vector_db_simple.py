#!/usr/bin/env python3
"""
Simple validation script for VectorDB LangChain implementation.
This script tests basic functionality without requiring the full test suite.
"""

import sys
import os
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from AIgnite.db.vector_db import VectorDB, VectorEntry

# Configure logging to see logger.info messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def test_basic_functionality():
    """Test basic VectorDB functionality."""
    print("Testing VectorDB implementation...")
    
    # Create test database
    test_db_path = "/data3/guofang/AIgnite-Solutions/AIgnite/test_simple_db/"
    try:
        vector_db = VectorDB(
            db_path=test_db_path,
            model_name='BAAI/bge-base-en-v1.5'
        )
        print("✓ VectorDB initialized successfully")
        
        # Test document data
        vector_db_doc_id = "001_test"
        test_text = "This is a test document about artificial intelligence and machine learning. It covers neural networks, deep learning, and natural language processing."
        test_metadata = {
            "doc_id": '001',
            "text_type": "document",
            "title": "Test AI Paper",
            "authors": ["Test Author"],
            "categories": ["cs.AI"]
        }
        
        # Test adding document
        success = vector_db.add_document(
            vector_db_id=vector_db_doc_id,
            text_to_emb=test_text,
            doc_metadata=test_metadata
        )
        if success:
            print("✓ Document added successfully")
        else:
            print("✗ Failed to add document")
            return False
        
        # Add 4 more test documents (2 sharing same doc_id)
        additional_docs = [
            {
                "vector_db_id": "002_test",
                "text": "This document discusses computer vision and image recognition techniques. It covers convolutional neural networks, object detection, and image classification methods.",
                "metadata": {
                    "doc_id": "002",
                    "text_type": "document",
                    "title": "Computer Vision Research",
                    "authors": ["Vision Researcher"],
                    "categories": ["cs.CV"]
                }
            },
            {
                "vector_db_id": "003_test",
                "text": "This is the first part of a comprehensive study on natural language processing. It introduces basic concepts and tokenization methods.",
                "metadata": {
                    "doc_id": "003",  # This doc_id will be shared
                    "text_type": "document",
                    "title": "NLP Study Part 1",
                    "authors": ["NLP Expert"],
                    "categories": ["cs.CL"]
                }
            },
            {
                "vector_db_id": "004_test",
                "text": "This is the second part of the natural language processing study. It covers advanced topics like transformers and attention mechanisms.",
                "metadata": {
                    "doc_id": "003",  # Same doc_id as previous document
                    "text_type": "document",
                    "title": "NLP Study Part 2",
                    "authors": ["NLP Expert"],
                    "categories": ["cs.CL"]
                }
            },
            {
                "vector_db_id": "005_test",
                "text": "This document explores reinforcement learning algorithms and their applications in game playing and robotics.",
                "metadata": {
                    "doc_id": "004",
                    "text_type": "document",
                    "title": "Reinforcement Learning Guide",
                    "authors": ["RL Specialist"],
                    "categories": ["cs.LG"]
                }
            }
        ]
        
        # Add additional documents
        for i, doc in enumerate(additional_docs, 2):
            success = vector_db.add_document(
                vector_db_id=doc["vector_db_id"],
                text_to_emb=doc["text"],
                doc_metadata=doc["metadata"]
            )
            if success:
                print(f"✓ Additional document {i} added successfully")
            else:
                print(f"✗ Failed to add additional document {i}")
                return False
        
        # Test saving
        if vector_db.save():
            print("✓ Database saved successfully")
        else:
            print("✗ Failed to save database")
            return False
        
        # Test searching
        print("\n--- Testing search functionality ---")
        results = vector_db.search("artificial intelligence", top_k=5)
        if results:
            print(f"✓ Search returned {len(results)} results")
            for i, (entry, score) in enumerate(results, 1):
                print(f"  {i}. doc_id: {entry.doc_id}, text_type: {entry.text_type}, score: {score:.4f}")
        else:
            print("✗ Search returned no results")
            return False
        
        # Test loading
        new_vector_db = VectorDB(
            db_path=test_db_path,
            model_name='BAAI/bge-base-en-v1.5'
        )
        if new_vector_db.exists():
            print("✓ Database loaded successfully")
        else:
            print("✗ Failed to load database")
            return False
        
        # Test search on loaded database
        print("\n--- Testing search on loaded database ---")
        loaded_results = new_vector_db.search("machine learning", top_k=5)
        if loaded_results:
            print(f"✓ Search on loaded database returned {len(loaded_results)} results")
            for i, (entry, score) in enumerate(loaded_results, 1):
                print(f"  {i}. doc_id: {entry.doc_id}, text_type: {entry.text_type}, score: {score:.4f}")
        else:
            print("✗ Search on loaded database returned no results")
            return False

        # Test doc_id filtering
        print("\n--- Testing doc_id filtering ---")
        
        # Test 1: Single doc_id filter
        print("Testing single doc_id filter (doc_id='001')...")
        filtered_results_1 = new_vector_db.search("machine learning", filters={"include": {"doc_ids": ["001"]}}, top_k=5)
        if filtered_results_1:
            print(f"✓ Single doc_id filter returned {len(filtered_results_1)} results")
            for i, (entry, score) in enumerate(filtered_results_1, 1):
                print(f"  {i}. doc_id: {entry.doc_id}, text_type: {entry.text_type}, score: {score:.4f}")
                if entry.doc_id != "001":
                    print(f"✗ Error: Found document with doc_id '{entry.doc_id}', expected only '001'")
                    return False
            print("✓ All results have correct doc_id '001'")
        else:
            print("✗ Single doc_id filter returned no results")
            return False
        
        # Test 2: Multiple doc_ids filter
        print("\nTesting multiple doc_ids filter (doc_ids=['001', '002'])...")
        filtered_results_2 = new_vector_db.search("neural networks", filters={"include": {"doc_ids": ["001", "002"]}}, top_k=5)
        if filtered_results_2:
            print(f"✓ Multiple doc_ids filter returned {len(filtered_results_2)} results")
            for i, (entry, score) in enumerate(filtered_results_2, 1):
                print(f"  {i}. doc_id: {entry.doc_id}, text_type: {entry.text_type}, score: {score:.4f}")
                if entry.doc_id not in ["001", "002"]:
                    print(f"✗ Error: Found document with doc_id '{entry.doc_id}', expected only '001' or '002'")
                    return False
            print("✓ All results have correct doc_ids ('001' or '002')")
        else:
            print("✗ Multiple doc_ids filter returned no results")
            return False
        
        # Test 3: Non-existent doc_id filter
        print("\nTesting non-existent doc_id filter (doc_id='999')...")
        filtered_results_3 = new_vector_db.search("artificial intelligence", filters={"include": {"doc_ids": ["999"]}}, top_k=5)
        if not filtered_results_3:
            print("✓ Non-existent doc_id filter correctly returned no results")
        else:
            print(f"✗ Error: Non-existent doc_id filter returned {len(filtered_results_3)} results, expected 0")
            return False
        
        # Test 4: Verify filtering accuracy by comparing with unfiltered search
        print("\nTesting filtering accuracy by comparing with unfiltered search...")
        unfiltered_results = new_vector_db.search("learning", top_k=10)
        filtered_results_4 = new_vector_db.search("learning", filters={"include": {"doc_ids": ["001", "004"]}}, top_k=10)
        
        if unfiltered_results and filtered_results_4:
            print(f"✓ Unfiltered search returned {len(unfiltered_results)} results")
            print(f"✓ Filtered search returned {len(filtered_results_4)} results")
            
            # Verify all filtered results have correct doc_ids
            for entry, score in filtered_results_4:
                if entry.doc_id not in ["001", "004"]:
                    print(f"✗ Error: Filtered result has doc_id '{entry.doc_id}', expected only '001' or '004'")
                    return False
            
            # Verify filtered results are subset of unfiltered results
            filtered_doc_ids = {entry.doc_id for entry, _ in filtered_results_4}
            unfiltered_doc_ids = {entry.doc_id for entry, _ in unfiltered_results}
            
            if filtered_doc_ids.issubset(unfiltered_doc_ids):
                print("✓ Filtered results are subset of unfiltered results")
            else:
                print("✗ Error: Filtered results contain doc_ids not in unfiltered results")
                return False
                
            print("✓ Doc_id filtering accuracy verified")
        else:
            print("✗ Failed to perform filtering accuracy test")
            return False

        # Test 5: Exclusion filters
        print("\n--- Testing doc_id exclusion filters ---")
        
        # Test 5.1: Exclude single doc_id
        print("Testing single doc_id exclusion filter (exclude doc_id='001')...")
        exclude_results_1 = new_vector_db.search("machine learning", filters={"exclude": {"doc_ids": ["001"]}}, top_k=10)
        if exclude_results_1:
            print(f"✓ Single doc_id exclusion filter returned {len(exclude_results_1)} results")
            for i, (entry, score) in enumerate(exclude_results_1, 1):
                print(f"  {i}. doc_id: {entry.doc_id}, text_type: {entry.text_type}, score: {score:.4f}")
                if entry.doc_id == "001":
                    print(f"✗ Error: Found excluded document with doc_id '001'")
                    return False
            print("✓ All results correctly exclude doc_id '001'")
        else:
            print("✗ Single doc_id exclusion filter returned no results")
            return False
        
        # Test 5.2: Exclude multiple doc_ids
        print("\nTesting multiple doc_ids exclusion filter (exclude doc_ids=['001', '002'])...")
        exclude_results_2 = new_vector_db.search("neural networks", filters={"exclude": {"doc_ids": ["001", "002"]}}, top_k=10)
        if exclude_results_2:
            print(f"✓ Multiple doc_ids exclusion filter returned {len(exclude_results_2)} results")
            for i, (entry, score) in enumerate(exclude_results_2, 1):
                print(f"  {i}. doc_id: {entry.doc_id}, text_type: {entry.text_type}, score: {score:.4f}")
                if entry.doc_id in ["001", "002"]:
                    print(f"✗ Error: Found excluded document with doc_id '{entry.doc_id}'")
                    return False
            print("✓ All results correctly exclude doc_ids '001' and '002'")
        else:
            print("✗ Multiple doc_ids exclusion filter returned no results")
            return False
        
        # Test 5.3: Exclude non-existent doc_id (should return all results)
        print("\nTesting exclusion of non-existent doc_id (exclude doc_id='999')...")
        exclude_results_3 = new_vector_db.search("artificial intelligence", filters={"exclude": {"doc_ids": ["999"]}}, top_k=10)
        if exclude_results_3:
            print(f"✓ Exclusion of non-existent doc_id returned {len(exclude_results_3)} results (expected all results)")
            print("✓ Correctly returned all results when excluding non-existent doc_id")
        else:
            print("✗ Exclusion of non-existent doc_id returned no results")
            return False
        
        # Test 5.4: Verify exclusion accuracy by comparing with unfiltered search
        print("\nTesting exclusion accuracy by comparing with unfiltered search...")
        unfiltered_results_exclude = new_vector_db.search("learning", top_k=10)
        exclude_results_4 = new_vector_db.search("learning", filters={"exclude": {"doc_ids": ["001", "002"]}}, top_k=10)
        
        if unfiltered_results_exclude and exclude_results_4:
            print(f"✓ Unfiltered search returned {len(unfiltered_results_exclude)} results")
            print(f"✓ Exclusion filtered search returned {len(exclude_results_4)} results")
            
            # Verify all excluded results don't have excluded doc_ids
            for entry, score in exclude_results_4:
                if entry.doc_id in ["001", "002"]:
                    print(f"✗ Error: Excluded result has doc_id '{entry.doc_id}', should be excluded")
                    return False
            
            # Verify excluded results are subset of unfiltered results
            excluded_doc_ids = {entry.doc_id for entry, _ in exclude_results_4}
            unfiltered_doc_ids = {entry.doc_id for entry, _ in unfiltered_results_exclude}
            
            if excluded_doc_ids.issubset(unfiltered_doc_ids):
                print("✓ Excluded results are subset of unfiltered results")
            else:
                print("✗ Error: Excluded results contain doc_ids not in unfiltered results")
                return False
                
            print("✓ Doc_id exclusion filtering accuracy verified")
        else:
            print("✗ Failed to perform exclusion filtering accuracy test")
            return False

        # Test deletion
        print("\n--- Testing document deletion ---")
        
        # Show all documents before deletion
        docs = new_vector_db.faiss_store.docstore._dict
        print("Documents before deletion:")
        for k, v in docs.items():
            print(f"  ID: {k}, doc_id: {v.metadata.get('doc_id', 'N/A')}, title: {v.metadata.get('title', 'N/A')}")
        
        # Test deleting a document with shared doc_id (should delete both documents with doc_id "003")
        print(f"\nDeleting documents with doc_id '003' (should delete 2 documents)...")
        if new_vector_db.delete_document("003"):
            print("✓ Documents with doc_id '003' deleted successfully")
        else:
            print("✗ Failed to delete documents with doc_id '003'")
            return False
        
        # Show remaining documents after deletion
        print("\nDocuments after deletion:")
        docs_after = new_vector_db.faiss_store.docstore._dict
        for k, v in docs_after.items():
            print(f"  ID: {k}, doc_id: {v.metadata.get('doc_id', 'N/A')}, title: {v.metadata.get('title', 'N/A')}")
        
        # Verify that documents with doc_id "003" are gone
        remaining_doc_ids = [v.metadata.get('doc_id') for v in docs_after.values()]
        if "003" not in remaining_doc_ids:
            print("✓ Confirmed: No documents with doc_id '003' remain")
        else:
            print("✗ Error: Documents with doc_id '003' still exist")
            return False
        
        print("\n🎉 All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {str(e)}")
        return False
    
    finally:
        # Cleanup
        try:
            if os.path.exists(f"{test_db_path}/index.pkl"):
                os.remove(f"{test_db_path}/index.pkl")
            if os.path.exists(f"{test_db_path}/index.faiss"):
                os.remove(f"{test_db_path}/index.faiss")
            print("✓ Test cleanup completed")
        except:
            pass

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
