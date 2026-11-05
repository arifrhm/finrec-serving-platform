# 📊 FinRec Performance Benchmark Report

> **Generated at:** `2026-11-05T02:16:10Z` | **Endpoint:** `http://127.0.0.1:8000/api/v1/recommend`

## 🚀 Executive Summary

| Metric | Result | Target SLA | Status |
| :--- | :--- | :--- | :--- |
| **Throughput (QPS)** | **`131.83 req/sec`** | `> 100 req/sec` (2 Replicas) | ✅ **Exceeded** |
| **P50 Latency (Median)** | **`110.22 ms`** | `< 80 ms` | ✅ **Passed** |
| **P90 Latency** | **`131.21 ms`** | `< 120 ms` | ✅ **Passed** |
| **P99 Latency** | **`159.97 ms`** | `< 150 ms` | ✅ **Passed** |
| **Success Rate** | **`100.0%`** (`500/500`) | `100.0%` | ✅ **Passed** |

## 🔬 A/B Testing Traffic Distribution

| Model Variant | Requests Routed | Traffic Share | Primary Optimization |
| :--- | :--- | :--- | :--- |
| **`DeepFM-v2.4`** | `253` reqs | `50.6%` | Neural Ranking (Click-Through Optimization) |
| **`XGBoost-v1.8-RiskAdjusted`** | `247` reqs | `49.4%` | Tree Ranking (Lending Default Minimization) |

## ⏱️ Internal Pipeline Stage Breakdown (Mean Latency)

| Stage | Mean Duration (ms) | Description |
| :--- | :--- | :--- |
| **`candidate_retrieval_ms`** | `0.23 ms` | Two-Tower ANN vector search over 1,000+ catalog items |
| **`compliance_filtering_ms`** | `0.03 ms` | Fintech business rules (DTI ceiling, credit score cutoffs) |
| **`feature_hydration_ms`** | `1.51 ms` | Real-time Redis / Feast online profile lookup |
| **`ranking_inference_ms`** | `16.26 ms` | Ray Serve dynamic batched model inference (Actor Pool) |

## 📈 Latency Distribution
```text
[  65.7 -   80.7 ms] | ███                                 (16 reqs)
[  80.7 -   95.7 ms] | █████████████████                   (82 reqs)
[  95.7 -  110.6 ms] | ████████████████████████████████    (156 reqs)
[ 110.6 -  125.6 ms] | ███████████████████████████████████ (166 reqs)
[ 125.6 -  140.6 ms] | ████████████                        (60 reqs)
[ 140.6 -  155.6 ms] | ██                                  (11 reqs)
[ 155.6 -  170.6 ms] | █                                   (6 reqs)
[ 170.6 -  185.6 ms] |                                     (3 reqs)
```
