import asyncio
from typing import Dict, Any
from src.services.data_generator import generate_dynamic_user_profile

class FeatureStoreClient:
    """
    Simulates production Feature Store integration (e.g. Feast on Redis Cluster).
    Retrieves low-latency user entity features and real-time streaming aggregates.
    Dynamically hydrates any user_id with realistic financial profile data.
    """
    def __init__(self):
        # Local hot cache simulating Redis MGET hit
        self._user_feature_cache: Dict[str, Dict[str, Any]] = {}

    async def get_online_features(self, user_id: str) -> Dict[str, Any]:
        """
        Simulates an asynchronous non-blocking Redis MGET / Feast online retrieval.
        Dynamically synthesizes and caches feature vectors for unknown/streaming user IDs.
        """
        # Emulate 1.2ms network I/O to Redis Cluster
        await asyncio.sleep(0.0012)
        
        if user_id not in self._user_feature_cache:
            profile = generate_dynamic_user_profile(user_id)
            self._user_feature_cache[user_id] = profile

        return self._user_feature_cache[user_id]
