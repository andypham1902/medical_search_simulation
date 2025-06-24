#!/usr/bin/env python3
"""
Test script for FAISS integration in Medical Search Simulation
"""

import sys
import os
import time
import requests
import json
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_api_health(base_url):
    """Test API health endpoint"""
    try:
        logger.info("Testing API health...")
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            logger.info(f"API Health Status: {health_data.get('status', 'unknown')}")
            
            # Check FAISS stats
            data_loaded = health_data.get('data_loaded', {})
            logger.info(f"Embeddings loaded: {data_loaded.get('embeddings_matrix', False)}")
            logger.info(f"Embeddings shape: {data_loaded.get('embeddings_shape', 'unknown')}")
            
            return True
        else:
            logger.error(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return False

def test_search_functionality(base_url, queries):
    """Test search functionality with FAISS"""
    logger.info("Testing search functionality...")
    
    results = []
    for query in queries:
        try:
            logger.info(f"Testing query: '{query}'")
            start_time = time.time()
            
            # Test without reranking
            payload = {
                "query": query,
                "use_reranker": False
            }
            
            response = requests.post(f"{base_url}/search", json=payload, timeout=30)
            search_time = time.time() - start_time
            
            if response.status_code == 200:
                search_results = response.json()
                num_results = len(search_results.get('results', []))
                logger.info(f"Query successful: {num_results} results in {search_time:.2f}s")
                
                # Test with reranking
                start_time = time.time()
                payload["use_reranker"] = True
                response = requests.post(f"{base_url}/search", json=payload, timeout=60)
                rerank_time = time.time() - start_time
                
                if response.status_code == 200:
                    rerank_results = response.json()
                    num_rerank_results = len(rerank_results.get('results', []))
                    logger.info(f"Reranking successful: {num_rerank_results} results in {rerank_time:.2f}s")
                    
                    results.append({
                        "query": query,
                        "search_time": search_time,
                        "rerank_time": rerank_time,
                        "num_results": num_results,
                        "num_rerank_results": num_rerank_results,
                        "success": True
                    })
                else:
                    logger.error(f"Reranking failed for query '{query}': {response.status_code}")
                    results.append({"query": query, "success": False, "error": "reranking_failed"})
            else:
                logger.error(f"Search failed for query '{query}': {response.status_code}")
                results.append({"query": query, "success": False, "error": "search_failed"})
                
        except Exception as e:
            logger.error(f"Error testing query '{query}': {e}")
            results.append({"query": query, "success": False, "error": str(e)})
    
    return results

def test_visit_functionality(base_url, test_urls):
    """Test visit functionality"""
    logger.info("Testing visit functionality...")
    
    for url in test_urls:
        try:
            logger.info(f"Testing visit: {url}")
            payload = {"url": url}
            
            start_time = time.time()
            response = requests.post(f"{base_url}/visit", json=payload, timeout=30)
            visit_time = time.time() - start_time
            
            if response.status_code == 200:
                visit_data = response.json()
                content_length = len(visit_data.get('data', ''))
                logger.info(f"Visit successful: {content_length} chars in {visit_time:.2f}s")
            else:
                logger.error(f"Visit failed for {url}: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error testing visit for {url}: {e}")

def run_performance_test(base_url, num_concurrent=10, num_queries=50):
    """Run concurrent performance test"""
    import concurrent.futures
    import threading
    
    logger.info(f"Running performance test: {num_concurrent} concurrent users, {num_queries} queries")
    
    test_queries = [
        "diabetes treatment",
        "cancer immunotherapy",
        "cardiovascular disease",
        "machine learning in medicine",
        "COVID-19 symptoms",
        "neural networks medical imaging",
        "drug discovery artificial intelligence",
        "clinical trials methodology"
    ] * (num_queries // 8 + 1)
    
    test_queries = test_queries[:num_queries]
    
    def worker(queries):
        results = []
        for query in queries:
            try:
                start_time = time.time()
                payload = {"query": query, "use_reranker": True}
                response = requests.post(f"{base_url}/search", json=payload, timeout=30)
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    search_results = response.json()
                    results.append({
                        "success": True,
                        "duration": duration,
                        "num_results": len(search_results.get('results', []))
                    })
                else:
                    results.append({"success": False, "duration": duration})
                    
            except Exception as e:
                results.append({"success": False, "error": str(e)})
        
        return results
    
    # Split queries among workers
    queries_per_worker = len(test_queries) // num_concurrent
    worker_queries = [test_queries[i*queries_per_worker:(i+1)*queries_per_worker] 
                     for i in range(num_concurrent)]
    
    # Add remaining queries to last worker
    if len(test_queries) % num_concurrent:
        worker_queries[-1].extend(test_queries[num_concurrent*queries_per_worker:])
    
    # Run concurrent test
    start_time = time.time()
    all_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        futures = [executor.submit(worker, queries) for queries in worker_queries]
        
        for future in concurrent.futures.as_completed(futures):
            all_results.extend(future.result())
    
    total_time = time.time() - start_time
    
    # Analyze results
    successful = [r for r in all_results if r.get('success', False)]
    failed = [r for r in all_results if not r.get('success', False)]
    
    if successful:
        avg_duration = sum(r['duration'] for r in successful) / len(successful)
        max_duration = max(r['duration'] for r in successful)
        min_duration = min(r['duration'] for r in successful)
        
        logger.info(f"Performance Test Results:")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Successful queries: {len(successful)}/{len(all_results)}")
        logger.info(f"  Average response time: {avg_duration:.2f}s")
        logger.info(f"  Min response time: {min_duration:.2f}s")
        logger.info(f"  Max response time: {max_duration:.2f}s")
        logger.info(f"  Throughput: {len(successful)/total_time:.2f} queries/sec")
    
    logger.info(f"Failed queries: {len(failed)}")

def main():
    parser = argparse.ArgumentParser(description="Test FAISS integration in Medical Search API")
    parser.add_argument("--url", default="http://localhost:10000", help="API base URL")
    parser.add_argument("--skip-health", action="store_true", help="Skip health check")
    parser.add_argument("--skip-search", action="store_true", help="Skip search tests")
    parser.add_argument("--skip-visit", action="store_true", help="Skip visit tests")
    parser.add_argument("--skip-performance", action="store_true", help="Skip performance tests")
    parser.add_argument("--concurrent", type=int, default=10, help="Number of concurrent users for performance test")
    parser.add_argument("--queries", type=int, default=50, help="Number of queries for performance test")
    
    args = parser.parse_args()
    
    logger.info(f"Testing Medical Search API at {args.url}")
    
    # Test queries for search functionality
    test_queries = [
        "diabetes treatment guidelines",
        "COVID-19 vaccine efficacy",
        "machine learning medical diagnosis",
        "cancer immunotherapy research",
        "cardiovascular disease prevention"
    ]
    
    # Test URLs for visit functionality (these should exist in the metadata)
    test_urls = [
        "https://pubmed.ncbi.nlm.nih.gov/12345678/",  # Example URLs
        "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1234567/"
    ]
    
    success = True
    
    # Health check
    if not args.skip_health:
        if not test_api_health(args.url):
            logger.error("Health check failed - API may not be ready")
            success = False
    
    # Search functionality
    if not args.skip_search and success:
        search_results = test_search_functionality(args.url, test_queries)
        failed_searches = [r for r in search_results if not r.get('success', False)]
        if failed_searches:
            logger.error(f"Search tests failed: {len(failed_searches)}/{len(search_results)}")
            success = False
        else:
            logger.info("All search tests passed!")
    
    # Visit functionality
    if not args.skip_visit and success:
        test_visit_functionality(args.url, test_urls)
    
    # Performance test
    if not args.skip_performance and success:
        run_performance_test(args.url, args.concurrent, args.queries)
    
    if success:
        logger.info("✅ All tests completed successfully!")
    else:
        logger.error("❌ Some tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()