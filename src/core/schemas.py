from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

class RiskTolerance(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"

class ProductCategory(str, Enum):
    PERSONAL_LOAN = "personal_loan"
    CREDIT_CARD = "credit_card"
    WEALTH_MANAGEMENT = "wealth_management"
    INSURANCE = "insurance"
    SAVINGS_ACCOUNT = "savings_account"

class RecommendationRequest(BaseModel):
    user_id: str = Field(..., description="Unique customer identifier")
    k: int = Field(default=5, ge=1, le=50, description="Top-K recommendations requested")
    categories: Optional[List[ProductCategory]] = Field(
        default=None, 
        description="Filter by specific product categories if requested by caller"
    )
    include_explanation: bool = Field(
        default=False, 
        description="Whether to generate LLM-powered compliance and advisory reasoning"
    )
    request_context: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Real-time context e.g. device, geo, active session intent"
    )

class ProductCandidate(BaseModel):
    product_id: str
    category: ProductCategory
    interest_rate_or_fee: float
    min_credit_score: int
    max_dti_ratio: float  # Debt-to-income maximum threshold
    base_embedding: Optional[List[float]] = None

class ScoredProduct(BaseModel):
    product_id: str
    category: ProductCategory
    score: float
    features_used: Dict[str, float] = Field(default_factory=dict)
    explanation: Optional[str] = None

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[ScoredProduct]
    pipeline_latency_ms: float
    stage_breakdown_ms: Dict[str, float]
    model_version: str
