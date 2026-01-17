from uuid import uuid4
from datetime import datetime
from src.application.dtos.request import AnalysisRequest
from src.application.dtos.response import AnalysisResponse
from src.config.settings import settings


class RunAnalysisUseCase:
    """Use case for running agent-based analysis."""

    def __init__(self, workflow_executor):
        self.workflow_executor = workflow_executor

    async def execute(self, request: AnalysisRequest) -> AnalysisResponse:
        """Execute the analysis workflow."""

        task_id = uuid4()
        started_at = datetime.utcnow()

        # Execute the workflow (simplified)
        result = await self.workflow_executor.run(
            query=request.query,
            context=request.context or {},
            max_iterations=request.max_iterations or settings.MAX_ITERATIONS,
        )

        return AnalysisResponse(
            task_id=task_id,
            status="completed",
            result=result.get("final_result"),
            iterations=len(result.get("intermediate_steps", [])),
            started_at=started_at,
            completed_at=datetime.utcnow(),
        )
