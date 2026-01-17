from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    """Request DTO for running an analysis."""

    query: str = Field(..., min_length=1, description="The analysis query")
    context: dict | None = Field(default=None, description="Additional context")
    max_iterations: int | None = Field(default=None, ge=1, le=20)
