import time
import numpy as np
from typing import List, Optional
from src.core.schemas import ProductCandidate, ProductCategory

# Mock in-memory financial product catalog representing diverse credit, wealth, and insurance offerings
MOCK_PRODUCT_CATALOG: List[ProductCandidate] = [
    ProductCandidate(product_id="LOAN-PREMIER-01", category=ProductCategory.PERSONAL_LOAN, interest_rate_or_fee=0.038, min_credit_score=720, max_dti_ratio=0.35),
    ProductCandidate(product_id="LOAN-STANDARD-02", category=ProductCategory.PERSONAL_LOAN, interest_rate_or_fee=0.065, min_credit_score=620, max_dti_ratio=0.45),
    ProductCandidate(product_id="CC-CASHBACK-PLAT", category=ProductCategory.CREDIT_CARD, interest_rate_or_fee=0.0, min_credit_score=680, max_dti_ratio=0.40),
    ProductCandidate(product_id="CC-MILES-SIGNATURE", category=ProductCategory.CREDIT_CARD, interest_rate_or_fee=120.0, min_credit_score=750, max_dti_ratio=0.30),
    ProductCandidate(product_id="WEALTH-INDEX-FUND", category=ProductCategory.WEALTH_MANAGEMENT, interest_rate_or_fee=0.002, min_credit_score=0, max_dti_ratio=1.0),
    ProductCandidate(product_id="WEALTH-FIXED-INCOME", category=ProductCategory.WEALTH_MANAGEMENT, interest_rate_or_fee=0.005, min_credit_score=0, max_dti_ratio=1.0),
    ProductCandidate(product_id="INS-TERM-LIFE-1M", category=ProductCategory.INSURANCE, interest_rate_or_fee=35.0, min_credit_score=0, max_dti_ratio=1.0),
    ProductCandidate(product_id="SAVINGS-HIGH-YIELD", category=ProductCategory.SAVINGS_ACCOUNT, interest_rate_or_fee=0.042, min_credit_score=0, max_dti_ratio=1.0),
]

class CandidateRetrievalService:
    """
    Candidate Generation / Retrieval Layer.
    Responsible for narrowing millions or thousands of catalog items down to Top-N candidates
    using vector similarity (Two-Tower Bi-Encoder / ANN vector index) in sub-5ms.
    """
    def __init__(self, embedding_dim: int = 64):
        self.dim = embedding_dim
        # Pre-generate deterministic embeddings for catalog items
        np.random.seed(42)
        self.catalog = MOCK_PRODUCT_CATALOG
        self.catalog_embeddings = np.random.randn(len(self.catalog), self.dim).astype(np.float32)
        # Normalize embeddings for cosine similarity
        self.catalog_embeddings /= np.linalg.norm(self.catalog_embeddings, axis=1, keepdims=True)

    def retrieve_candidates(
        self, 
        user_embedding: np.ndarray, 
        allowed_categories: Optional[List[ProductCategory]] = None, 
        top_k: int = 50
    ) -> List[ProductCandidate]:
        """
        Executes ANN / cosine similarity search against pre-indexed catalog embeddings.
        """
        # Cosine dot product
        scores = np.dot(self.catalog_embeddings, user_embedding)
        ranked_indices = np.argsort(scores)[::-1]

        candidates: List[ProductCandidate] = []
        for idx in ranked_indices:
            candidate = self.catalog[idx]
            if allowed_categories and candidate.category not in allowed_categories:
                continue
            candidates.append(candidate)
            if len(candidates) >= top_k:
                break
                
        return candidates
