#!/usr/bin/env python3
import asyncio
import aiohttp
import time
import random
import argparse
import json
import sys
from typing import List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import statistics

@dataclass
class RequestMetrics:
    endpoint: str
    status_code: int
    response_time: float
    timestamp: float
    error: str = None
    response_size: int = 0

@dataclass
class TestResults:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_time: float = 0
    requests_per_second: float = 0
    response_times: List[float] = field(default_factory=list)
    errors: Dict[str, int] = field(default_factory=dict)
    status_codes: Dict[int, int] = field(default_factory=dict)
    
    def add_metric(self, metric: RequestMetrics):
        self.total_requests += 1
        if metric.status_code == 200:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            
        if metric.error:
            self.errors[metric.error] = self.errors.get(metric.error, 0) + 1
            
        self.status_codes[metric.status_code] = self.status_codes.get(metric.status_code, 0) + 1
        self.response_times.append(metric.response_time)
    
    def calculate_stats(self):
        if self.response_times:
            return {
                "min": min(self.response_times),
                "max": max(self.response_times),
                "mean": statistics.mean(self.response_times),
                "median": statistics.median(self.response_times),
                "p95": statistics.quantiles(self.response_times, n=20)[18] if len(self.response_times) > 20 else max(self.response_times),
                "p99": statistics.quantiles(self.response_times, n=100)[98] if len(self.response_times) > 100 else max(self.response_times)
            }
        return {}

class StressTest:
    def __init__(self, base_url: str, ccu: int = 64):
        self.base_url = base_url.rstrip('/')
        self.ccu = ccu
        self.search_queries = [
            "cancer treatment",
            "covid-19 vaccine",
            "diabetes management",
            "heart disease prevention",
            "alzheimer's research",
            "antibiotics resistance",
            "mental health therapy",
            "gene therapy CRISPR",
            "immunotherapy breakthrough",
            "stem cell research",
            "neurological disorders",
            "infectious disease control",
            "precision medicine",
            "drug discovery AI",
            "clinical trials phase 3"
        ]
        self.sample_urls = []
        self.search_results = TestResults()
        self.visit_results = TestResults()
    
    async def run_preflight_checks(self):
        """Run pre-flight checks to ensure API is healthy before stress testing"""
        print("🚀 Running Pre-flight Checks")
        print("=" * 80)
        
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Check health endpoint
            print("🔍 Testing health check...")
            try:
                async with session.get(f"{self.base_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        print("✅ Health check passed")
                        print(f"   Models loaded: {data.get('models_loaded', 'Unknown')}")
                    else:
                        print(f"❌ Health check failed: HTTP {response.status}")
                        return False
            except aiohttp.ClientConnectorError:
                print(f"❌ Could not connect to API at {self.base_url}")
                print("   Make sure the API is running")
                return False
            except Exception as e:
                print(f"❌ Health check error: {e}")
                return False
            
            # Test search endpoint
            print("\n🔍 Testing search endpoint...")
            test_query = "diabetes treatment"
            try:
                payload = {"query": test_query, "top_k": 5}
                async with session.post(f"{self.base_url}/search", json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        results_count = len(data.get("results", []))
                        print(f"✅ Search endpoint working - found {results_count} results")
                        
                        # Store URLs for visit test
                        if results_count > 0:
                            for result in data["results"][:3]:
                                if result.get("url"):
                                    self.sample_urls.append(result["url"])
                            first_result = data["results"][0]
                            print(f"   Top result: {first_result.get('metadata', {}).get('paper_title', 'N/A')[:60]}...")
                    else:
                        print(f"❌ Search failed: HTTP {response.status}")
                        return False
            except Exception as e:
                print(f"❌ Search test error: {e}")
                return False
            
            # Test visit endpoint
            print("\n🔍 Testing visit endpoint...")
            if self.sample_urls:
                test_url = self.sample_urls[0]
                try:
                    payload = {"url": test_url}
                    async with session.post(f"{self.base_url}/visit", json=payload) as response:
                        if response.status == 200:
                            data = await response.json()
                            content_length = len(data.get("data", ""))
                            print(f"✅ Visit endpoint working - retrieved {content_length} characters")
                        else:
                            print(f"❌ Visit failed: HTTP {response.status}")
                            return False
                except Exception as e:
                    print(f"❌ Visit test error: {e}")
                    return False
            else:
                print("⚠️  No URLs available to test visit endpoint")
            
            print("\n✅ All pre-flight checks passed!")
            print("=" * 80)
            return True
        
    async def make_search_request(self, session: aiohttp.ClientSession, query: str) -> RequestMetrics:
        endpoint = f"{self.base_url}/search"
        start_time = time.time()
        
        try:
            payload = {
                "query": query,
                "top_k": 10,
                "rerank": random.choice([True, False])
            }
            
            async with session.post(endpoint, json=payload) as response:
                response_data = await response.text()
                response_time = time.time() - start_time
                
                metric = RequestMetrics(
                    endpoint="/search",
                    status_code=response.status,
                    response_time=response_time,
                    timestamp=start_time,
                    response_size=len(response_data)
                )
                
                if response.status == 200:
                    try:
                        data = json.loads(response_data)
                        if data.get("results"):
                            for result in data["results"][:3]:
                                if result.get("url"):
                                    self.sample_urls.append(result["url"])
                    except:
                        pass
                        
                return metric
                
        except asyncio.TimeoutError:
            return RequestMetrics(
                endpoint="/search",
                status_code=0,
                response_time=time.time() - start_time,
                timestamp=start_time,
                error="Timeout"
            )
        except Exception as e:
            return RequestMetrics(
                endpoint="/search",
                status_code=0,
                response_time=time.time() - start_time,
                timestamp=start_time,
                error=str(type(e).__name__)
            )
    
    async def make_visit_request(self, session: aiohttp.ClientSession, url: str) -> RequestMetrics:
        endpoint = f"{self.base_url}/visit"
        start_time = time.time()
        
        try:
            payload = {"url": url}
            
            async with session.post(endpoint, json=payload) as response:
                response_data = await response.text()
                response_time = time.time() - start_time
                
                return RequestMetrics(
                    endpoint="/visit",
                    status_code=response.status,
                    response_time=response_time,
                    timestamp=start_time,
                    response_size=len(response_data)
                )
                
        except asyncio.TimeoutError:
            return RequestMetrics(
                endpoint="/visit",
                status_code=0,
                response_time=time.time() - start_time,
                timestamp=start_time,
                error="Timeout"
            )
        except Exception as e:
            return RequestMetrics(
                endpoint="/visit",
                status_code=0,
                response_time=time.time() - start_time,
                timestamp=start_time,
                error=str(type(e).__name__)
            )
    
    async def user_simulation(self, session: aiohttp.ClientSession, user_id: int, duration: int):
        """Simulate a single user making requests"""
        end_time = time.time() + duration
        
        while time.time() < end_time:
            # 70% search, 30% visit
            if random.random() < 0.7 or not self.sample_urls:
                query = random.choice(self.search_queries)
                metric = await self.make_search_request(session, query)
                self.search_results.add_metric(metric)
            else:
                url = random.choice(self.sample_urls[-100:])  # Use recent URLs
                metric = await self.make_visit_request(session, url)
                self.visit_results.add_metric(metric)
            
            # Random delay between requests (0.5 to 2 seconds)
            await asyncio.sleep(random.uniform(0.5, 2.0))
    
    async def run_stress_test(self, duration: int = 60, skip_preflight: bool = False):
        """Run the stress test with specified CCU"""
        # Run pre-flight checks first unless skipped
        if not skip_preflight:
            if not await self.run_preflight_checks():
                print("\n❌ Pre-flight checks failed. Please fix the issues and try again.")
                return False
            
            # Wait a moment for models to stabilize after checks
            print("\n⏳ Waiting 2 seconds before starting stress test...")
            await asyncio.sleep(2)
        
        print(f"\n🔥 Starting stress test with {self.ccu} concurrent users for {duration} seconds")
        print(f"Target: {self.base_url}")
        print("-" * 80)
        
        timeout = aiohttp.ClientTimeout(total=30)
        connector = aiohttp.TCPConnector(limit=self.ccu * 2)
        
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            # Warm up with a few requests
            print("Warming up...")
            for i in range(3):
                query = self.search_queries[i]
                await self.make_search_request(session, query)
            
            print(f"Starting main test at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            start_time = time.time()
            
            # Create tasks for all concurrent users
            tasks = [
                self.user_simulation(session, user_id, duration)
                for user_id in range(self.ccu)
            ]
            
            # Run all users concurrently
            await asyncio.gather(*tasks)
            
            total_time = time.time() - start_time
            
        # Calculate final metrics
        self.search_results.total_time = total_time
        self.search_results.requests_per_second = self.search_results.total_requests / total_time
        
        self.visit_results.total_time = total_time
        self.visit_results.requests_per_second = self.visit_results.total_requests / total_time
        
        self.print_results()
        return True
    
    def print_results(self):
        print("\n" + "=" * 80)
        print("STRESS TEST RESULTS")
        print("=" * 80)
        
        # Search endpoint results
        print("\n/search Endpoint:")
        print("-" * 40)
        print(f"Total Requests: {self.search_results.total_requests}")
        print(f"Successful: {self.search_results.successful_requests}")
        print(f"Failed: {self.search_results.failed_requests}")
        print(f"Success Rate: {self.search_results.successful_requests / max(1, self.search_results.total_requests) * 100:.2f}%")
        print(f"Requests/sec: {self.search_results.requests_per_second:.2f}")
        
        search_stats = self.search_results.calculate_stats()
        if search_stats:
            print(f"\nResponse Times (ms):")
            print(f"  Min: {search_stats['min']*1000:.2f}")
            print(f"  Max: {search_stats['max']*1000:.2f}")
            print(f"  Mean: {search_stats['mean']*1000:.2f}")
            print(f"  Median: {search_stats['median']*1000:.2f}")
            print(f"  95th percentile: {search_stats['p95']*1000:.2f}")
            print(f"  99th percentile: {search_stats['p99']*1000:.2f}")
        
        if self.search_results.errors:
            print(f"\nErrors:")
            for error, count in self.search_results.errors.items():
                print(f"  {error}: {count}")
        
        # Visit endpoint results
        print("\n/visit Endpoint:")
        print("-" * 40)
        print(f"Total Requests: {self.visit_results.total_requests}")
        print(f"Successful: {self.visit_results.successful_requests}")
        print(f"Failed: {self.visit_results.failed_requests}")
        print(f"Success Rate: {self.visit_results.successful_requests / max(1, self.visit_results.total_requests) * 100:.2f}%")
        print(f"Requests/sec: {self.visit_results.requests_per_second:.2f}")
        
        visit_stats = self.visit_results.calculate_stats()
        if visit_stats:
            print(f"\nResponse Times (ms):")
            print(f"  Min: {visit_stats['min']*1000:.2f}")
            print(f"  Max: {visit_stats['max']*1000:.2f}")
            print(f"  Mean: {visit_stats['mean']*1000:.2f}")
            print(f"  Median: {visit_stats['median']*1000:.2f}")
            print(f"  95th percentile: {visit_stats['p95']*1000:.2f}")
            print(f"  99th percentile: {visit_stats['p99']*1000:.2f}")
        
        if self.visit_results.errors:
            print(f"\nErrors:")
            for error, count in self.visit_results.errors.items():
                print(f"  {error}: {count}")
        
        # Combined stats
        print("\n" + "=" * 80)
        print("COMBINED STATISTICS")
        print("=" * 80)
        total_requests = self.search_results.total_requests + self.visit_results.total_requests
        total_successful = self.search_results.successful_requests + self.visit_results.successful_requests
        print(f"Total Requests: {total_requests}")
        print(f"Total Successful: {total_successful}")
        print(f"Overall Success Rate: {total_successful / max(1, total_requests) * 100:.2f}%")
        print(f"Total Requests/sec: {total_requests / self.search_results.total_time:.2f}")
        print(f"Concurrent Users: {self.ccu}")

def main():
    parser = argparse.ArgumentParser(description="Stress test for medical search API with pre-flight checks")
    parser.add_argument("--url", default="http://localhost:10000", help="Base URL of the API")
    parser.add_argument("--ccu", type=int, default=64, help="Concurrent users (default: 64)")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds (default: 60)")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip pre-flight checks")
    
    args = parser.parse_args()
    
    print("🚀 Medical Search API Stress Test")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  API URL: {args.url}")
    print(f"  Concurrent Users: {args.ccu}")
    print(f"  Test Duration: {args.duration} seconds")
    print(f"  Skip Pre-flight: {args.skip_preflight}")
    print("=" * 80)
    
    stress_test = StressTest(args.url, args.ccu)
    success = asyncio.run(stress_test.run_stress_test(args.duration, args.skip_preflight))
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()