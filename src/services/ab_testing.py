import hashlib
from typing import Dict, Any, Tuple

class ExperimentationRouter:
    """
    Production A/B Testing & Model Traffic Allocation Engine.
    Deterministically routes users to experiment variants using consistent hashing.
    """
    def __init__(self):
        # 50/50 split between Model A (DeepFM neural ranking) and Model B (XGBoost risk-weighted)
        self.experiments: Dict[str, Dict[str, Any]] = {
            "rec_model_scoring_v2": {
                "variants": [
                    {"name": "DeepFM-v2.4", "weight": 50, "features": "deep_cross_features"},
                    {"name": "XGBoost-v1.8-RiskAdjusted", "weight": 50, "features": "tree_binned_features"}
                ]
            }
        }

    def assign_variant(self, user_id: str, experiment_name: str = "rec_model_scoring_v2") -> Tuple[str, str]:
        """
        Calculates a deterministic 0-99 bucket using SHA256 of user_id + experiment_name.
        Returns (variant_name, feature_profile)
        """
        hash_input = f"{experiment_name}:{user_id}".encode("utf-8")
        hash_val = int(hashlib.sha256(hash_input).hexdigest()[:8], 16)
        bucket = hash_val % 100

        exp = self.experiments.get(experiment_name)
        if not exp:
            return ("DeepFM-v2.4", "default")

        cumulative = 0
        for variant in exp["variants"]:
            cumulative += variant["weight"]
            if bucket < cumulative:
                return (variant["name"], variant["features"])

        return (exp["variants"][0]["name"], "default")
