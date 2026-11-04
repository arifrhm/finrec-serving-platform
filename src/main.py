import ray
from ray import serve
from src.services.ranking import RankingScorerDeployment
from src.services.pipeline import FinRecPipelineDeployment

# 1. Bind Child Deployments (Actor Pool)
ranking_scorer = RankingScorerDeployment.bind()

# 2. Bind Ingress Pipeline Orchestrator with the child handle
entrypoint = FinRecPipelineDeployment.bind(ranking_scorer)

if __name__ == "__main__":
    # Local standalone execution
    ray.init(ignore_reinit_error=True)
    serve.start(http_options={"host": "0.0.0.0", "port": 8000})
    serve.run(entrypoint)
    print("🚀 FinRec Serving Engine listening on http://0.0.0.0:8000")
