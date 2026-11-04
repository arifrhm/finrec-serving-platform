import argparse
import time
import random
import statistics
import threading
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple

DEFAULT_URL = "http://127.0.0.1:8000/api/v1/recommend"
CATEGORIES = ["personal_loan", "credit_card", "wealth_management", "insurance", "savings_account"]

thread_local = threading.local()

def get_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=50, pool_maxsize=50)
        thread_local.session.mount("http://", adapter)
    return thread_local.session

def render_ascii_histogram(latencies: List[float], bins: int = 8):
    """Generates an ASCII latency distribution histogram for terminal display."""
    if not latencies:
        return
    min_val, max_val = min(latencies), max(latencies)
    if min_val == max_val:
        return
    step = (max_val - min_val) / bins
    counts = [0] * bins

    for lat in latencies:
        idx = min(int((lat - min_val) / step), bins - 1)
        counts[idx] += 1

    max_count = max(counts) if max(counts) > 0 else 1
    print("\n📊 Latency Distribution Histogram (ms):")
    for i in range(bins):
        low = min_val + i * step
        high = low + step
        bar_len = int((counts[i] / max_count) * 35)
        bar = "█" * bar_len
        print(f"  [{low:6.1f} - {high:6.1f} ms] | {bar:<35} ({counts[i]} reqs)")

def send_dynamic_request(target_url: str, request_id: int, include_llm: bool, random_category: bool) -> Tuple[bool, float, str, Dict[str, float]]:
    session = get_session()
    # Dynamic synthetic persona
    user_id = f"user_stream_{random.randint(1000, 9999)}"
    
    payload: Dict[str, Any] = {
        "user_id": user_id,
        "k": 5,
        "include_explanation": include_llm,
        "request_context": {
            "session_id": f"sess_{request_id}",
            "intent": random.choice(["apply_credit", "grow_wealth", "explore_savings"])
        }
    }
    if random_category and random.random() > 0.5:
        payload["categories"] = [random.choice(CATEGORIES)]

    t0 = time.perf_counter()
    try:
        resp = session.post(target_url, json=payload, timeout=8.0)
        dur_ms = (time.perf_counter() - t0) * 1000
        if resp.status_code == 200:
            data = resp.json()
            return (True, dur_ms, data.get("model_version", "unknown"), data.get("stage_breakdown_ms", {}))
        return (False, dur_ms, "error", {})
    except Exception:
        dur_ms = (time.perf_counter() - t0) * 1000
        return (False, dur_ms, "timeout", {})

def execute_load_test(target_url: str, total_requests: int, concurrency: int, include_llm: bool, random_category: bool):
    print("\n" + "=" * 65)
    print("      🚀 FINREC DYNAMIC LOAD TEST & A/B EXPERIMENTATION")
    print("=" * 65)
    print(f"Target Endpoint       : {target_url}")
    print(f"Total Requests        : {total_requests}")
    print(f"Concurrent Workers    : {concurrency}")
    print(f"Dynamic User Personas : Enabled (randomized profiles & credit scores)")
    print(f"LLM Reasoning Step    : {'Enabled (vLLM / FlashAttention)' if include_llm else 'Disabled (Ranking only)'}")
    print(f"Category Filtering    : {'Dynamic random filter' if random_category else 'Full catalog (All 5 cats)'}")
    print("-" * 65)

    latencies: List[float] = []
    variant_counts: Dict[str, int] = {}
    successes = 0
    failures = 0
    stage_aggregates: Dict[str, List[float]] = {}

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_dynamic_request, target_url, i, include_llm, random_category)
            for i in range(total_requests)
        ]
        for f in futures:
            ok, lat, variant, stages = f.result()
            latencies.append(lat)
            if ok:
                successes += 1
                variant_counts[variant] = variant_counts.get(variant, 0) + 1
                for s_name, s_ms in stages.items():
                    stage_aggregates.setdefault(s_name, []).append(s_ms)
            else:
                failures += 1

    total_time = time.perf_counter() - t_start
    qps = total_requests / total_time

    latencies.sort()
    p50 = statistics.median(latencies)
    p90 = latencies[int(len(latencies) * 0.90)]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]

    print("\n📈 PERFORMANCE & LATENCY METRICS:")
    print(f"  Completed Requests  : {total_requests} (Success: {successes}, Failed: {failures})")
    print(f"  Throughput (QPS)    : {qps:.2f} requests/sec")
    print(f"  Total Wall Time     : {total_time:.2f} seconds")
    print(f"  Latency P50 (Median): {p50:.2f} ms")
    print(f"  Latency P90         : {p90:.2f} ms")
    print(f"  Latency P95         : {p95:.2f} ms")
    print(f"  Latency P99         : {p99:.2f} ms")
    print(f"  Min / Max Latency   : {min(latencies):.2f} ms / {max(latencies):.2f} ms")

    print("\n🔬 A/B TESTING TRAFFIC ALLOCATION:")
    for variant, count in sorted(variant_counts.items()):
        pct = (count / successes) * 100 if successes else 0
        print(f"  • {variant:<30}: {count:4d} reqs ({pct:5.1f}%)")

    if stage_aggregates:
        print("\n⏱️  INTERNAL PIPELINE STAGE BREAKDOWN (Mean):")
        for stage, times in sorted(stage_aggregates.items()):
            mean_stage = statistics.mean(times)
            print(f"  - {stage:<25}: {mean_stage:6.2f} ms")

    render_ascii_histogram(latencies)
    print("=" * 65 + "\n")

def main():
    parser = argparse.ArgumentParser(description="FinRec Dynamic Performance Benchmark Harness")
    parser.add_argument("--url", default=DEFAULT_URL, help="Target recommendation endpoint")
    parser.add_argument("--requests", type=int, default=300, help="Total requests to dispatch")
    parser.add_argument("--concurrency", type=int, default=15, help="Number of concurrent worker threads")
    parser.add_argument("--include-llm", action="store_true", help="Enable LLM reasoning step")
    parser.add_argument("--random-category", action="store_true", help="Simulate random category query filters")
    args = parser.parse_args()

    execute_load_test(
        target_url=args.url,
        total_requests=args.requests,
        concurrency=args.concurrency,
        include_llm=args.include_llm,
        random_category=args.random_category
    )

if __name__ == "__main__":
    main()
