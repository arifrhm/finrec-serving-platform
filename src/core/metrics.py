from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "finrec_recommendation_requests_total",
    "Total number of recommendation requests served",
    ["status", "endpoint"]
)

PIPELINE_LATENCY = Histogram(
    "finrec_pipeline_latency_seconds",
    "Total end-to-end pipeline latency in seconds",
    ["model_version"],
    buckets=[0.005, 0.010, 0.020, 0.035, 0.050, 0.075, 0.100, 0.250, 0.500]
)

STAGE_LATENCY = Histogram(
    "finrec_stage_latency_seconds",
    "Latency broken down by individual pipeline stage",
    ["stage"],
    buckets=[0.001, 0.003, 0.005, 0.010, 0.015, 0.025, 0.050]
)

DYNAMIC_BATCH_SIZE = Histogram(
    "finrec_ranking_dynamic_batch_size",
    "Distribution of batch sizes processed by Ray Serve ranking deployment",
    buckets=[1, 2, 4, 8, 16, 32, 64, 128]
)

FALLBACK_TRIGGERED = Counter(
    "finrec_fallback_triggered_total",
    "Number of times cold-start / rule-based fallback was triggered",
    ["reason"]
)
