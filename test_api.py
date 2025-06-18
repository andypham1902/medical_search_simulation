#!/usr/bin/env python3
"""
Example script to test the Medical Search Simulation API
"""

import requests
import json
import time

API_BASE_URL = "http://192.168.0.11:8000"


def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed")
            print(f"   Models loaded: {data['models_loaded']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure it's running on localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


def test_search():
    """Test the search endpoint"""
    print("\n🔍 Testing search endpoint...")

    test_queries = [
        "diabetes treatment",
        "cancer therapy",
        "cardiovascular disease",
        "machine learning medical",
    ]

    for query in test_queries:
        print(f"   Searching for: '{query}'")
        try:
            response = requests.post(
                f"{API_BASE_URL}/search", json={"query": query}, timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                results_count = len(data.get("results", []))
                print(f"   ✅ Found {results_count} results")

                # Show first result if available
                if results_count > 0:
                    first_result = data["results"][0]
                    print(
                        f"      Top result: {first_result.get('metadata', {}).get('paper_title', 'N/A')}"
                    )
                    print(f"      Score: {first_result.get('score', 'N/A')}")
            else:
                print(f"   ❌ Search failed: {response.status_code} - {response.text}")

        except requests.exceptions.Timeout:
            print(f"   ⏰ Search timeout for query: '{query}'")
        except Exception as e:
            print(f"   ❌ Search error for '{query}': {e}")


def test_visit():
    """Test the visit endpoint"""
    print("\n🔍 Testing visit endpoint...")

    # First, get a URL from search results
    try:
        search_response = requests.post(
            f"{API_BASE_URL}/search", json={"query": "test query"}, timeout=30
        )

        if search_response.status_code == 200:
            search_data = search_response.json()
            results = search_data.get("results", [])

            if results:
                test_url = results[0]["url"]
                print(f"   Visiting URL: {test_url}")

                visit_response = requests.post(
                    f"{API_BASE_URL}/visit", json={"url": test_url}, timeout=30
                )

                if visit_response.status_code == 200:
                    visit_data = visit_response.json()
                    passages_count = len(visit_data.get("data", "").split())
                    print(f"   ✅ Found {passages_count} words")

                    # Show first passage if available
                    if passages_count > 0:
                        first_passage = visit_data["data"][0]
                        print(f"      First passage preview: {first_passage[:100]}...")
                else:
                    print(
                        f"   ❌ Visit failed: {visit_response.status_code} - {visit_response.url}"
                    )
            else:
                print("   ⚠️  No search results to test visit with")
        else:
            print(
                f"   ❌ Could not get search results for visit test: {search_response.status_code}"
            )

    except Exception as e:
        print(f"   ❌ Visit test error: {e}")


def main():
    print("🚀 Starting API Test Suite")
    print("=" * 50)

    # Test health check first
    if not test_health_check():
        print("\n❌ API is not healthy. Please check the server.")
        return

    # Wait a moment for models to be fully loaded
    print("\n⏳ Waiting for models to finish loading...")
    time.sleep(2)

    # Test search functionality
    test_search()

    # Test visit functionality
    test_visit()

    print("\n🎉 Test suite completed!")
    print("\nTo run the API manually:")
    print("  python api.py")
    print("\nTo test individual endpoints:")
    print(
        f"  curl -X POST {API_BASE_URL}/search -H 'Content-Type: application/json' -d '{{\"query\": \"your query\"}}'"
    )
    print(
        f"  curl -X POST {API_BASE_URL}/visit -H 'Content-Type: application/json' -d '{{\"url\": \"your_url\"}}'"
    )


if __name__ == "__main__":
    main()
