import time
import statistics
import threading
import requests
from concurrent.futures import ThreadPoolExecutor

TARGET_URL = "http://127.0.0.1:8000/api/v1/recommend"
TOTAL_REQUESTS = 500
CONCURRENCY = 20

PAYLOAD = {
    "user_id": "user_premium_001",
    "k": 5,
    "include_explanation": False
}

thread_local = threading.local()

def get_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
        thread_local.session.mount("http://", adapter)
    return thread_local.session

def send_request(session_id: int):
    session = get_session()
    t0 = time.perf_counter()
    try:
        resp = session.post(TARGET_URL, json=PAYLOAD, timeout=5.0)
        latency_ms = (time.perf_counter() - t0) * 1000
        return (resp.status_code == 200, latency_ms)
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        return (False, latency_ms)

def run_load_test():
    print(f"🚀 Commencing Load Test against {TARGET_URL}")
    print(f"Parameters: Total={TOTAL_REQUESTS} requests, Concurrency={CONCURRENCY} workers\n")

    latencies = []
    successes = 0
    failures = 0

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = [executor.submit(send_request, i) for i in range(TOTAL_REQUESTS)]
        for f in futures:
            ok, lat = f.result()
            latencies.append(lat)
            if ok:
                successes += 1
            else:
                failures += 1

    total_time = time.perf_counter() - t_start
    qps = TOTAL_REQUESTS / total_time

    latencies.sort()
    p50 = statistics.median(latencies)
    p90 = latencies[int(len(latencies) * 0.90)]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]

    print("=" * 55)
    print("           FINREC BENCHMARK SUMMARY")
    print("=" * 55)
    print(f"Total Completed Requests: {TOTAL_REQUESTS}")
    print(f"Successful (200 OK)     : {successes}")
    print(f"Failed / Dropped        : {failures}")
    print(f"Throughput (QPS)        : {qps:.2f} req/sec")
    print(f"Total Duration          : {total_time:.2f} s")
    print("-" * 55)
    print(f"Latency P50 (Median)    : {p50:.2f} ms")
    print(f"Latency P90             : {p90:.2f} ms")
    print(f"Latency P95             : {p95:.2f} ms")
    print(f"Latency P99             : {p99:.2f} ms")
    print(f"Min / Max               : {min(latencies):.2f} ms / {max(latencies):.2f} ms")
    print("=" * 55)

if __name__ == "__main__":
    run_load_test()
