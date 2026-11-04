from typing import List, Dict, Any
from src.core.schemas import ScoredProduct, ProductCandidate, ProductCategory

class FintechComplianceFilter:
    """
    Post-Ranking Business Rules and Regulatory Compliance Guardrails.
    Ensures credit risk compliance, anti-predatory lending laws, and catalog diversity.
    """
    def filter_and_diversify(
        self,
        scored_products: List[ScoredProduct],
        catalog_lookup: Dict[str, ProductCandidate],
        user_features: Dict[str, Any],
        top_k: int = 5
    ) -> List[ScoredProduct]:
        user_credit = user_features.get("credit_score", 0)
        user_dti = user_features.get("dti_ratio", 1.0)
        days_since_apply = user_features.get("days_since_last_application", 999)

        eligible: List[ScoredProduct] = []
        category_counts: Dict[ProductCategory, int] = {}

        for sp in scored_products:
            meta = catalog_lookup.get(sp.product_id)
            if not meta:
                continue

            # Hard Rule 1: Credit Score Hard Floor (Regulatory / Underwriting risk)
            if user_credit < meta.min_credit_score:
                continue

            # Hard Rule 2: DTI Ratio (Avoid over-indebtedness)
            if user_dti > meta.max_dti_ratio:
                continue

            # Hard Rule 3: Cooling-off Period (Don't pitch loans if recent application pending)
            if meta.category == ProductCategory.PERSONAL_LOAN and days_since_apply < 30:
                continue

            # Soft Rule 4: Catalog Diversification (Max 2 items from the same category)
            current_cat_count = category_counts.get(meta.category, 0)
            if current_cat_count >= 2:
                continue

            category_counts[meta.category] = current_cat_count + 1
            eligible.append(sp)

            if len(eligible) >= top_k:
                break

        return eligible
