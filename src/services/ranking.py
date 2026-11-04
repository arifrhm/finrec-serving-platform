import time
from typing import List, Dict, Any, Tuple
import numpy as np
from ray import serve
from src.core.schemas import ProductCandidate, ScoredProduct
from src.core.metrics import DYNAMIC_BATCH_SIZE, STAGE_LATENCY

@serve.deployment(
    name="RankingScorerDeployment",
    num_replicas=2,
    max_ongoing_requests=100,
    ray_actor_options={"num_cpus": 1.0}
)
class RankingScorerDeployment:
    """
    Ray Serve deployment implementing Dynamic Batching for ML model scoring (e.g. DeepFM / DLRM / XGBoost).
    
    Dynamic batching merges parallel incoming prediction tasks within a 5ms window,
    maximizing vectorized inference efficiency while maintaining strict SLA constraints.
    """
    def __init__(self):
        # Simulated pre-warmed model weights / ONNX runtime session
        np.random.seed(1337)
        self.feature_weights = np.array([0.35, -0.25, 0.20, 0.15, -0.10], dtype=np.float32)
        print("[RankingScorerDeployment] Model loaded into Ray Actor memory.")

    @serve.batch(max_batch_size=64, batch_wait_timeout_s=0.005)
    async def score_batch(
        self, 
        items_batch: List[Tuple[Dict[str, Any], List[ProductCandidate]]]
    ) -> List[List[ScoredProduct]]:
        """
        Dynamically batched scoring method. Ray Serve runtime groups calls from different callers.
        """
        batch_size = len(items_batch)
        DYNAMIC_BATCH_SIZE.observe(batch_size)
        start_t = time.perf_counter()

        all_results: List[List[ScoredProduct]] = []

        # Vectorized scoring across all pairs in the batch
        for user_features, candidates in items_batch:
            credit_score_norm = (user_features.get("credit_score", 650) - 300) / 550.0
            dti_norm = 1.0 - min(user_features.get("dti_ratio", 0.3), 1.0)
            income_log = np.log1p(user_features.get("monthly_income", 5000.0)) / 10.0
            
            scored_candidates: List[ScoredProduct] = []
            for candidate in candidates:
                # Feature vector: [credit_score_norm, dti_norm, income_log, rate_incentive, risk_penalty]
                rate_incentive = 1.0 - min(candidate.interest_rate_or_fee, 1.0)
                risk_penalty = 1.0 if candidate.min_credit_score > user_features.get("credit_score", 650) else 0.0

                x = np.array([credit_score_norm, dti_norm, income_log, rate_incentive, risk_penalty], dtype=np.float32)
                # Compute dot product + sigmoid probability
                raw_logit = float(np.dot(self.feature_weights, x))
                prob_score = 1.0 / (1.0 + np.exp(-raw_logit))

                scored_candidates.append(
                    ScoredProduct(
                        product_id=candidate.product_id,
                        category=candidate.category,
                        score=round(prob_score, 4),
                        features_used={
                            "credit_norm": round(float(credit_score_norm), 3),
                            "dti_norm": round(float(dti_norm), 3)
                        }
                    )
                )
            
            # Sort individual user's candidates by descending ML score
            scored_candidates.sort(key=lambda s: s.score, reverse=True)
            all_results.append(scored_candidates)

        duration = time.perf_counter() - start_t
        STAGE_LATENCY.labels(stage="ranking_batch").observe(duration)
        return all_results

    async def __call__(self, user_features: Dict[str, Any], candidates: List[ProductCandidate]) -> List[ScoredProduct]:
        return await self.score_batch((user_features, candidates))
