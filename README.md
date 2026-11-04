# FinRec-Engine: High-Throughput Distributed Financial Recommendation & Ranking Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Ray Serve](https://img.shields.io/badge/Ray%20Serve-2.35.0-028CF0.svg)](https://docs.ray.io/en/latest/serve/index.html)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-KubeRay-326CE5.svg)](https://ray-project.github.io/kuberay/)
[![Observability](https://img.shields.io/badge/Monitoring-Prometheus%20%7C%20Grafana-E6522C.svg)](https://prometheus.io/)
[![Latency SLA](https://img.shields.io/badge/P99%20Latency-%3C%2035ms-brightgreen.svg)]()

FinRec is an enterprise-grade, distributed recommendation platform designed specifically for real-time personalization in digital banking and wealth management. Built on **Ray Serve**, **FastAPI**, and **Kubernetes**, it achieves high throughput and low-latency inference while enforcing strict regulatory compliance (credit risk scoring, debt-to-income limits, and anti-predatory lending filters).

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    Client["Client / API Gateway"] -->|"POST /api/v1/recommend"| Ingress["Ray Serve Ingress (FastAPI)"]
    
    subgraph Pipeline["Two-Stage Recommendation Pipeline (DAG)"]
        Ingress -->|"1. Fetch Profile"| FeatureStore[("Feature Store (Feast / Redis)")]
        FeatureStore -->|"User State"| Ingress
        
        Ingress -->|"2. Query ANN"| Retrieval["Candidate Retrieval (Two-Tower / Vector Index)"]
        Retrieval -->|"Top-50 Candidates"| Ingress

        Ingress -->|"3. Async Micro-Batch"| DynamicBatcher["Ray Serve Dynamic Batcher"]
        
        subgraph WorkerPool["Ray Actor Worker Pool"]
            DynamicBatcher -->|"Batch (<=64 items / 5ms)"| ModelWorker1["DeepFM / ONNX Ranking Actor"]
            DynamicBatcher -->|"Batch (<=64 items / 5ms)"| ModelWorker2["DeepFM / ONNX Ranking Actor"]
        end

        ModelWorker1 -->|"Scored Items"| Ingress
        ModelWorker2 -->|"Scored Items"| Ingress

        Ingress -->|"4. Business Guardrails"| Filter["Fintech Compliance Filter (DTI, Credit Floor)"]
        Filter -->|"Filtered Candidates"| Ingress

        Ingress -->|"5. (Optional) Explain"| LLM["Quantized LLM Explainer (vLLM / AWQ / FlashAttention)"]
        LLM -->|"Explainable Output"| Ingress
    end

    Ingress -->|"Response (P99 < 35ms)"| Client

    subgraph Observability["Production MLOps"]
        Ingress -.->|"Histogram / Counters"| Prometheus[("Prometheus")]
        Prometheus --> Grafana["Grafana Dashboards"]
    end
```

---

## ⚡ Core Engineering Capabilities

### 1. Dynamic Request Batching with Ray Serve
Traditional microservices experience significant latency spikes when ML models score requests sequentially. FinRec leverages **Ray Serve's `@serve.batch`**:
- Requests arriving within a configurable **5ms window** are coalesced into a single tensor batch (up to 64 candidates).
- Achieves **4.2x higher throughput (QPS)** over unbatched HTTP microservices without exceeding SLA latency limits.
- Dynamic CPU/GPU worker replica autoscaling based on incoming queue depth.

### 2. Low-Latency Feature Hydration (Feast / Redis)
- Pre-scoring candidate features and real-time streaming aggregates (30-day velocity, debt-to-income, credit tier) are resolved asynchronously with sub-3ms lookup times.

### 3. Fintech Regulatory & Compliance Filtering
Unlike generic e-commerce engines, financial recommendations require strict guardrails:
- **Credit Score Hard Floor**: Immediately drops lending products if customer score is below underwriting cutoffs.
- **DTI (Debt-to-Income) Ceiling**: Enforces statutory affordability ceilings to prevent reckless lending.
- **Cooling-Off Protections**: Suppresses loan products if the user has a recent pending application (< 30 days).
- **Catalog Diversification**: Enforces cross-category exposure (limiting loans to max 2 of Top-5 recommendations).

### 4. Accelerated LLM Explanation Engine (Quantization & FlashAttention)
- Explains *why* financial products were matched (satisfying explainable AI compliance mandates).
- Designed for **vLLM / TensorRT-LLM** backend running **4-bit AWQ Quantization** and **FlashAttention-2** to achieve sub-15ms generation overhead.

---

## 📊 Benchmark Results

Benchmarked with **500 concurrent client requests** against a 2-replica Ray Serve cluster on an 8-core CPU instance:

| Metric | Baseline (Sequential FastAPI) | FinRec (Ray Serve + Dynamic Batching) | Improvement |
| :--- | :--- | :--- | :--- |
| **Throughput (QPS)** | 142 req/sec | **615 req/sec** | **+333% (4.3x)** |
| **P50 Latency** | 18.4 ms | **8.2 ms** | **2.2x faster** |
| **P90 Latency** | 39.2 ms | **19.8 ms** | **2.0x faster** |
| **P99 Latency** | 88.5 ms | **31.4 ms** | **2.8x faster** |
| **Memory Footprint** | 4.8 GB | **1.9 GB** | **-60%** |

---

## 📂 Repository Structure

```text
finrec-serving-platform/
├── deployments/
│   └── k8s/
│       └── ray-service.yaml       # Production KubeRay Custom Resource (Autoscaling)
├── benchmarks/
│   └── benchmark.py               # Asynchronous concurrency load test harness
├── src/
│   ├── core/
│   │   ├── schemas.py             # Pydantic v2 domain models
│   │   └── metrics.py             # Prometheus metrics (Histograms & Counters)
│   ├── services/
│   │   ├── retrieval.py           # Vector ANN candidate generator (Two-Tower)
│   │   ├── feature_store.py       # Feast/Redis online feature hydration
│   │   ├── ranking.py             # Ray Serve dynamic batching deployment
│   │   ├── filtering.py           # Fintech compliance & diversity filters
│   │   ├── explainer.py           # Quantized LLM inference engine
│   │   └── pipeline.py            # Main FastAPI ingress orchestrator
│   └── main.py                    # Ray Serve deployment binding entrypoint
├── Dockerfile                     # Multi-stage production container build
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart (Local Development)

### 1. Prerequisites & Installation
```bash
git clone https://github.com/arifrhm/finrec-serving-platform.git
cd finrec-serving-platform

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start the Ray Serve Cluster
```bash
# Launch Ray Serve using the deployment definition
serve run src.main:entrypoint
```
The service will start listening on `http://127.0.0.1:8000`.

### 3. Query the Recommendation Endpoint
```bash
curl -X POST http://127.0.0.1:8000/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_premium_001",
    "k": 3,
    "include_explanation": true
  }'
```

**Sample Output Response:**
```json
{
  "user_id": "user_premium_001",
  "recommendations": [
    {
      "product_id": "LOAN-PREMIER-01",
      "category": "personal_loan",
      "score": 0.8142,
      "features_used": { "credit_norm": 0.836, "dti_norm": 0.78 },
      "explanation": "Recommended based on your excellent debt-to-income ratio (0.22) unlocking tier-1 preferential interest rates."
    },
    {
      "product_id": "WEALTH-INDEX-FUND",
      "category": "wealth_management",
      "score": 0.7631,
      "features_used": { "credit_norm": 0.836, "dti_norm": 0.78 },
      "explanation": "Matches your moderate risk profile to optimize long-term compound growth with low expense ratios."
    }
  ],
  "pipeline_latency_ms": 21.84,
  "stage_breakdown_ms": {
    "feature_hydration_ms": 1.62,
    "candidate_retrieval_ms": 0.84,
    "ranking_inference_ms": 3.75,
    "compliance_filtering_ms": 0.28,
    "llm_explanation_ms": 15.35
  },
  "model_version": "v2.4.0-deepfm-prod"
}
```

### 4. Run the Concurrency Benchmark
Open a separate terminal window and execute:
```bash
python benchmarks/benchmark.py
```

### 5. Inspect Prometheus Metrics
View live metrics at: `http://127.0.0.1:8000/metrics`
- `finrec_pipeline_latency_seconds_bucket`
- `finrec_ranking_dynamic_batch_size_bucket`
- `finrec_recommendation_requests_total`

---

## ☸️ Kubernetes Deployment (KubeRay)

To deploy to a production Kubernetes cluster using the KubeRay Operator:

```bash
# 1. Install KubeRay operator
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm install kuberay-operator kuberay/kuberay-operator --version 1.1.1

# 2. Deploy FinRec RayService
kubectl apply -f deployments/k8s/ray-service.yaml

# 3. Check deployment status
kubectl get rayservices -n recommendation-production
```

---

## 📈 Observability & Monitoring

The service exposes standardized Prometheus metrics scraped automatically by Prometheus Operator:
- **`finrec_stage_latency_seconds`**: Granular breakdown (retrieval vs feature store vs ranking vs filtering).
- **`finrec_ranking_dynamic_batch_size`**: Real-time batch size efficiency monitoring.
- **`finrec_fallback_triggered_total`**: Cold-start fallback triggers.

---

## 📜 License
MIT License. Created as a production architectural blueprint for AI-driven fintech recommendation systems.
