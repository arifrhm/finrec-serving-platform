import asyncio
from typing import Dict, Any, List

class FeatureStoreClient:
    """
    Simulates production Feature Store integration (e.g. Feast on Redis Cluster).
    Retrieves low-latency user entity features and real-time streaming aggregates.
    """
    def __init__(self):
        # Simulated hot-cache in Redis
        self._user_feature_cache: Dict[str, Dict[str, Any]] = {
            "user_premium_001": {
                "credit_score": 760,
                "dti_ratio": 0.22,
                "monthly_income": 12500.0,
                "avg_monthly_spend_30d": 4200.0,
                "risk_tolerance": "moderate",
                "days_since_last_application": 180,
                "preferred_category": "wealth_management",
            },
            "user_standard_002": {
                "credit_score": 640,
                "dti_ratio": 0.38,
                "monthly_income": 4500.0,
                "avg_monthly_spend_30d": 1800.0,
                "risk_tolerance": "low",
                "days_since_last_application": 15,
                "preferred_category": "credit_card",
            },
        }
        # Fallback profile for cold-start / unprofiled users
        self._default_profile: Dict[str, Any] = {
            "credit_score": 650,
            "dti_ratio": 0.30,
            "monthly_income": 5000.0,
            "avg_monthly_spend_30d": 2000.0,
            "risk_tolerance": "moderate",
            "days_since_last_application": 999,
            "preferred_category": "savings_account",
        }

    async def get_online_features(self, user_id: str) -> Dict[str, Any]:
        """
        Simulates an asynchronous non-blocking Redis MGET / Feast online retrieval.
        """
        # Emulate 1.5ms network I/O to Redis Cluster
        await asyncio.sleep(0.0015)
        return self._user_feature_cache.get(user_id, self._default_profile)
