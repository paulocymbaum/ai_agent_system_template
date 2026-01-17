from abc import ABC, abstractmethod
from uuid import UUID
from src.domain.agents.entities import AgentEntity, TaskEntity


class AgentRepository(ABC):
    """Port for agent persistence."""

    @abstractmethod
    async def save(self, agent: AgentEntity) -> AgentEntity:
        pass

    @abstractmethod
    async def find_by_id(self, agent_id: UUID) -> AgentEntity | None:
        pass


class EmbeddingService(ABC):
    """Port for embedding generation."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        pass

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        pass
