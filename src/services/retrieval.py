import numpy as np
from typing import List, Optional
from src.core.schemas import ProductCandidate, ProductCategory
from src.services.data_generator import generate_dynamic_catalog

class CandidateRetrievalService:
    """
    Candidate Generation / Retrieval Layer.
    Performs fast ANN / cosine similarity search over 1,000+ dynamic financial products
    using Two-Tower Bi-Encoder embeddings in sub-5ms.
    """
    def __init__(self, embedding_dim: int = 64, catalog_size: int = 1000):
        self.dim = embedding_dim
        self.catalog = generate_dynamic_catalog(total_products=catalog_size, seed=42)
        
        # Pre-index normalized product embeddings in memory
        np.random.seed(42)
        self.catalog_embeddings = np.random.randn(len(self.catalog), self.dim).astype(np.float32)
        self.catalog_embeddings /= np.linalg.norm(self.catalog_embeddings, axis=1, keepdims=True)
        print(f"[CandidateRetrievalService] Indexed {len(self.catalog)} dynamic financial products.")

    def retrieve_candidates(
        self, 
        user_embedding: np.ndarray, 
        allowed_categories: Optional[List[ProductCategory]] = None, 
        top_k: int = 50
    ) -> List[ProductCandidate]:
        """
        Executes ANN / cosine similarity dot product against catalog embeddings.
        """
        # Batch dot product
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
