import random
import numpy as np
from typing import List, Dict, Any
from src.core.schemas import ProductCandidate, ProductCategory

# Curated catalog templates for realistic financial product generation
CATALOG_TEMPLATES = {
    ProductCategory.PERSONAL_LOAN: [
        {"prefix": "LOAN-PREMIER", "rate_range": (0.028, 0.045), "min_credit": (720, 780), "dti": 0.35},
        {"prefix": "LOAN-STANDARD", "rate_range": (0.055, 0.089), "min_credit": (600, 680), "dti": 0.45},
        {"prefix": "LOAN-FASTCASH", "rate_range": (0.099, 0.150), "min_credit": (500, 590), "dti": 0.55},
        {"prefix": "LOAN-DEBTCONSOL", "rate_range": (0.049, 0.075), "min_credit": (640, 710), "dti": 0.50},
    ],
    ProductCategory.CREDIT_CARD: [
        {"prefix": "CC-PLATINUM-CASHBACK", "rate_range": (0.0, 0.0), "min_credit": (680, 740), "dti": 0.40},
        {"prefix": "CC-SIGNATURE-MILES", "rate_range": (150.0, 300.0), "min_credit": (730, 800), "dti": 0.30},
        {"prefix": "CC-STUDENT-STARTER", "rate_range": (0.0, 25.0), "min_credit": (0, 600), "dti": 0.60},
        {"prefix": "CC-BUSINESS-REWARDS", "rate_range": (80.0, 200.0), "min_credit": (700, 760), "dti": 0.35},
    ],
    ProductCategory.WEALTH_MANAGEMENT: [
        {"prefix": "WEALTH-INDEX-ETF", "rate_range": (0.0015, 0.0035), "min_credit": (0, 0), "dti": 1.0},
        {"prefix": "WEALTH-GLOBAL-TECH", "rate_range": (0.0045, 0.0085), "min_credit": (0, 0), "dti": 1.0},
        {"prefix": "WEALTH-ESG-SUSTAINABLE", "rate_range": (0.0030, 0.0060), "min_credit": (0, 0), "dti": 1.0},
        {"prefix": "WEALTH-FIXED-INCOME-BOND", "rate_range": (0.0020, 0.0040), "min_credit": (0, 0), "dti": 1.0},
    ],
    ProductCategory.INSURANCE: [
        {"prefix": "INS-TERM-LIFE-1M", "rate_range": (25.0, 60.0), "min_credit": (0, 0), "dti": 1.0},
        {"prefix": "INS-CRITICAL-ILLNESS", "rate_range": (40.0, 95.0), "min_credit": (0, 0), "dti": 1.0},
        {"prefix": "INS-COMPREHENSIVE-HEALTH", "rate_range": (50.0, 120.0), "min_credit": (0, 0), "dti": 1.0},
    ],
    ProductCategory.SAVINGS_ACCOUNT: [
        {"prefix": "SAVINGS-HIGH-YIELD-4.5", "rate_range": (0.038, 0.046), "min_credit": (0, 0), "dti": 1.0},
        {"prefix": "SAVINGS-DIGITAL-FLEXI", "rate_range": (0.025, 0.035), "min_credit": (0, 0), "dti": 1.0},
        {"prefix": "SAVINGS-TIME-DEPOSIT-12M", "rate_range": (0.032, 0.041), "min_credit": (0, 0), "dti": 1.0},
    ],
}

def generate_dynamic_catalog(total_products: int = 1000, seed: int = 42) -> List[ProductCandidate]:
    """
    Generates a dynamic, large-scale financial product catalog with randomized yet realistic parameters.
    """
    random.seed(seed)
    np.random.seed(seed)
    catalog: List[ProductCandidate] = []

    categories = list(ProductCategory)
    for i in range(total_products):
        cat = categories[i % len(categories)]
        template = random.choice(CATALOG_TEMPLATES[cat])
        
        rate = round(random.uniform(*template["rate_range"]), 4)
        min_credit = random.randint(*template["min_credit"]) if template["min_credit"][1] > 0 else 0
        dti = template["dti"]

        product_id = f"{template['prefix']}-{i+1:04d}"
        catalog.append(ProductCandidate(
            product_id=product_id,
            category=cat,
            interest_rate_or_fee=rate,
            min_credit_score=min_credit,
            max_dti_ratio=dti
        ))

    return catalog

def generate_dynamic_user_profile(user_id: str) -> Dict[str, Any]:
    """
    Generates a deterministic yet highly dynamic user financial persona based on user_id hash.
    Simulates real-time credit score, debt-to-income, income, spend patterns, and risk tiers.
    """
    # Deterministic hash to guarantee consistent profile per user_id during testing
    user_hash = abs(hash(user_id))
    rnd = random.Random(user_hash)

    # Realistic Gaussian credit score distribution centered at 685 (range 350 - 850)
    credit_score = int(np.clip(rnd.gauss(685, 80), 350, 850))
    # Debt-to-income ratio (range 0.10 - 0.65)
    dti_ratio = round(np.clip(rnd.gauss(0.32, 0.12), 0.10, 0.65), 3)
    # Monthly income ($2,500 - $35,000)
    monthly_income = round(float(np.exp(rnd.uniform(7.8, 10.4))), 2)
    # Spend volume (20% - 60% of monthly income)
    spend_30d = round(monthly_income * rnd.uniform(0.20, 0.60), 2)
    # Days since last application
    days_since_apply = rnd.choice([5, 12, 28, 45, 90, 180, 365, 999])

    risk_tolerance = rnd.choice(["low", "moderate", "high"])
    preferred_category = rnd.choice(list(ProductCategory)).value

    return {
        "user_id": user_id,
        "credit_score": credit_score,
        "dti_ratio": dti_ratio,
        "monthly_income": monthly_income,
        "avg_monthly_spend_30d": spend_30d,
        "days_since_last_application": days_since_apply,
        "risk_tolerance": risk_tolerance,
        "preferred_category": preferred_category,
    }
