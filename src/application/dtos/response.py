from datetime import datetime
from pydantic import BaseModel, Field
from uuid import UUID


class AnalysisResponse(BaseModel):
    """Response DTO for analysis results."""

    task_id: UUID
    status: str
    result: str | None = None
    iterations: int
    started_at: datetime
    completed_at: datetime | None = None
