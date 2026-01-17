from langgraph.graph import StateGraph, END
from src.domain.workflows.state import AgentState
from src.infrastructure.graphs.nodes import router_node, analysis_node, finalize_node


def create_workflow() -> StateGraph:
    """Create the LangGraph workflow."""

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("finalize", finalize_node)

    # Define edges
    workflow.set_entry_point("router")
    workflow.add_edge("router", "analysis")
    workflow.add_edge("analysis", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()
