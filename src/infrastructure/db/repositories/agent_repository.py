from src.domain.agents.entities import AgentEntity


class AgentRepository:
    """Simple placeholder repository interface. Replace with SQLAlchemy implementation."""

    async def save(self, agent: AgentEntity) -> AgentEntity:
        return agent

    async def find_by_id(self, agent_id):
        return None
