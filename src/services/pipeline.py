import time
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from ray import serve
from ray.serve.handle import DeploymentHandle

from src.core.schemas import RecommendationRequest, RecommendationResponse
from src.core.metrics import REQUEST_COUNT, PIPELINE_LATENCY, STAGE_LATENCY
from src.services.retrieval import CandidateRetrievalService, MOCK_PRODUCT_CATALOG
from src.services.feature_store import FeatureStoreClient
from src.services.filtering import FintechComplianceFilter
from src.services.explainer import QuantizedLLMExplainer

app = FastAPI(
    title="FinRec Serving Platform",
    description="High-Throughput Distributed Financial Recommendation & Ranking Engine",
    version="1.0.0"
)

@serve.deployment(
    name="FinRecPipelineDeployment",
    num_replicas=2,
    max_ongoing_requests=200,
    ray_actor_options={"num_cpus": 1.0}
)
@serve.ingress(app)
class FinRecPipelineDeployment:
    """
    Main Orchestrator Deployment.
    Receives HTTP traffic, coordinates the end-to-end RecSys DAG, and interfaces
    with the Ray Serve dynamic batching ranking deployment.
    """
    def __init__(self, ranking_handle: DeploymentHandle):
        self.ranking_handle = ranking_handle
        self.retrieval_service = CandidateRetrievalService(embedding_dim=64)
        self.feature_store = FeatureStoreClient()
        self.compliance_filter = FintechComplianceFilter()
        self.llm_explainer = QuantizedLLMExplainer()
        self.catalog_map = {item.product_id: item for item in MOCK_PRODUCT_CATALOG}
        self.model_version = "v2.4.0-deepfm-prod"

    @app.get("/healthz")
    async def healthcheck(self):
        return {"status": "healthy", "service": "finrec-engine", "version": self.model_version}

    @app.get("/metrics")
    async def metrics():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/api/v1/recommend", response_model=RecommendationResponse)
    async def recommend(self, req: RecommendationRequest):
        t0 = time.perf_counter()
        stage_times = {}

        try:
            # 1. Real-time Feature Hydration (Feast / Redis)
            t_feat_start = time.perf_counter()
            user_features = await self.feature_store.get_online_features(req.user_id)
            stage_times["feature_hydration_ms"] = (time.perf_counter() - t_feat_start) * 1000
            STAGE_LATENCY.labels(stage="hydration").observe(stage_times["feature_hydration_ms"] / 1000)

            # 2. Candidate Retrieval (Vector / Bi-Encoder Two-Tower)
            t_ret_start = time.perf_counter()
            # Deterministic user embedding vector for simulation
            np.random.seed(abs(hash(req.user_id)) % (2**31))
            user_vec = np.random.randn(64).astype(np.float32)
            user_vec /= np.linalg.norm(user_vec)

            candidates = self.retrieval_service.retrieve_candidates(
                user_embedding=user_vec,
                allowed_categories=req.categories,
                top_k=50
            )
            stage_times["candidate_retrieval_ms"] = (time.perf_counter() - t_ret_start) * 1000
            STAGE_LATENCY.labels(stage="retrieval").observe(stage_times["candidate_retrieval_ms"] / 1000)

            if not candidates:
                raise HTTPException(status_code=404, detail="No eligible candidates retrieved.")

            # 3. Dynamic Batched ML Ranking (Ray Serve Actor Pool)
            t_rank_start = time.perf_counter()
            # Asynchronous RPC to dynamically batched Ray Actor
            scored_candidates = await self.ranking_handle.remote(user_features, candidates)
            stage_times["ranking_inference_ms"] = (time.perf_counter() - t_rank_start) * 1000

            # 4. Fintech Compliance & Diversification Guardrails
            t_filt_start = time.perf_counter()
            final_recs = self.compliance_filter.filter_and_diversify(
                scored_products=scored_candidates,
                catalog_lookup=self.catalog_map,
                user_features=user_features,
                top_k=req.k
            )
            stage_times["compliance_filtering_ms"] = (time.perf_counter() - t_filt_start) * 1000

            # 5. Optional LLM Explanation (vLLM / TensorRT accelerated)
            if req.include_explanation and final_recs:
                t_llm_start = time.perf_counter()
                final_recs = await self.llm_explainer.generate_explanations(req.user_id, final_recs)
                stage_times["llm_explanation_ms"] = (time.perf_counter() - t_llm_start) * 1000

            total_latency = (time.perf_counter() - t0) * 1000
            PIPELINE_LATENCY.labels(model_version=self.model_version).observe(total_latency / 1000)
            REQUEST_COUNT.labels(status="200", endpoint="/recommend").inc()

            return RecommendationResponse(
                user_id=req.user_id,
                recommendations=final_recs,
                pipeline_latency_ms=round(total_latency, 2),
                stage_breakdown_ms={k: round(v, 2) for k, v in stage_times.items()},
                model_version=self.model_version
            )

        except Exception as e:
            REQUEST_COUNT.labels(status="500", endpoint="/recommend").inc()
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
