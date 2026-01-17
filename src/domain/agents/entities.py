from datetime import datetime
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class AgentEntity(BaseModel):
    """Domain entity representing an AI agent."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    role: str
    model: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        frozen = False


class TaskEntity(BaseModel):
    """Domain entity representing a task."""

    id: UUID = Field(default_factory=uuid4)
    agent_id: UUID
    description: str
    status: str = "pending"
    result: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
