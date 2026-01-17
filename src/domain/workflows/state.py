from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from operator import add


class AgentState(TypedDict):
    """State definition for LangGraph workflow."""

    messages: Annotated[Sequence[BaseMessage], add]
    current_agent: str
    task_description: str
    intermediate_steps: list[dict]
    final_result: str | None
