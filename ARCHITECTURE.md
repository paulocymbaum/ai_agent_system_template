# Architecture & System Design: AI Agent Backend (Enhanced)

## 1. Overview
This document outlines the architectural standards for a scalable Python backend designed to orchestrate LLMs and autonomous agents. It emphasizes strict separation of concerns using DDD and integrates specialized libraries for robustness and data processing.

### Core Tech Stack
*   **Language:** Python 3.11+
*   **Orchestration:** LangChain & LangGraph
*   **Web Framework:** FastAPI (Async)
*   **Database:** PostgreSQL (Data) + Redis (State/Cache)

### Extended Library Ecosystem
To support production-grade agents, we integrate the following:

1.  **Data & Validation:** `pydantic` (v2), `pydantic-settings`.
2.  **Resilience:** `tenacity` (Retries for flaky LLM APIs).
3.  **Local ML & Embeddings:** `sentence-transformers`, `transformers`, `torch` (for local RAG/Embeddings).
4.  **Calculation & Analysis:** `pandas`, `numpy` (for Agent Tools).
5.  **HTTP Client:** `httpx` (Async requests).

---

## 2. Architectural Pattern: Hexagonal (Ports & Adapters)

We adhere to **Clean Architecture**. The dependency rule remains: **Inner layers define interfaces; Outer layers implement them.**

### Layer Breakdown with Libraries

1.  **Domain Layer (The Core)**
    *   **Libraries Allowed:** `pydantic` (for schemas/validation), `uuid`, `datetime`.
    *   **Role:** Defines Entities, Value Objects, and Repository Interfaces.
    *   **Note:** No heavy ML libraries (`torch`, `pandas`) here.

2.  **Application Layer (The Orchestrator)**
    *   **Libraries Allowed:** `pydantic`, `tenacity` (for retry policies on use cases).
    *   **Role:** Use Cases, DTOs, Service Interfaces.

3.  **Infrastructure Layer (The Implementer)**
    *   **Libraries Allowed:** `langchain`, `langgraph`, `sqlalchemy`, `transformers`, `sentence-transformers`, `pandas`, `numpy`, `httpx`.
    *   **Role:**
        *   **LLM/Graph:** Implements LangGraph nodes.
        *   **ML:** Loads HuggingFace models for local embeddings.
        *   **Tools:** Uses Pandas/NumPy to perform calculations requested by agents.

4.  **Presentation Layer (The Entry Point)**
    *   **Libraries Allowed:** `fastapi`, `pydantic-settings`.
    *   **Role:** HTTP handling, Environment configuration.

---

## 3. File Structure

We expand the structure to accommodate ML models and calculation tools.

```text
src/
├── config/
│   └── settings.py         # Pydantic Settings (Env vars)
│
├── domain/                 # LAYER 1: Pure Python
│   ├── agents/
│   │   ├── entities.py     # Pydantic models for Domain Entities
│   │   └── ports.py        # Interfaces (e.g., IEmbeddingModel)
│   └── workflows/
│       └── state.py        # TypedDict/Pydantic State definitions
│
├── application/            # LAYER 2: Orchestration
│   ├── dtos/               # Input/Output models
│   └── use_cases/
│       └── run_analysis.py # Orchestrates the flow
│
├── infrastructure/         # LAYER 3: Heavy Lifting
│   ├── db/
│   │   └── repositories/   # SQL Implementations
│   │
│   ├── llm/
│   │   ├── client.py       # OpenAI/Anthropic wrappers
│   │   └── resilience.py   # Tenacity decorators
│   │
│   ├── ml/                 # Local ML (Transformers)
│   │   └── embeddings.py   # Uses sentence-transformers
│   │
│   ├── tools/              # Agent Tools (Calculations)
│   │   ├── data_tools.py   # Uses Pandas/NumPy
│   │   └── web_tools.py    # Uses Httpx
│   │
│   └── graphs/             # LangGraph Definitions
│       ├── nodes.py
│       └── workflow.py
│
└── main.py                 # FastAPI App
```

---

## 4. Implementation Guidelines & Library Usage

### 4.1. Configuration (`pydantic-settings`)
Do not use `os.getenv` directly. Use a centralized settings object.

```python
# src/config/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    DATABASE_URL: str
    # Local ML config
    EMBEDDING_MODEL_PATH: str = "all-MiniLM-L6-v2" 

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
```

### 4.2. Resilience (`tenacity`)
LLMs and external APIs fail often. Implement retries in the **Infrastructure** layer.

```python
# src/infrastructure/llm/resilience.py
from tenacity import retry, stop_after_attempt, wait_exponential

# Decorator for LLM calls
robust_llm_call = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
```

### 4.3. Local ML (`transformers` / `sentence-transformers`)
If using local embeddings to save costs or for privacy, implement the Domain Interface in Infrastructure.

**Domain Interface:**
```python
# src/domain/agents/ports.py
from abc import ABC, abstractmethod

class EmbeddingService(ABC):
    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        pass
```

**Infrastructure Implementation:**
```python
# src/infrastructure/ml/embeddings.py
from sentence_transformers import SentenceTransformer
from src.domain.agents.ports import EmbeddingService

class LocalHuggingFaceEmbeddings(EmbeddingService):
    def __init__(self, model_name: str):
        # Load model only once (Singleton pattern recommended via DI)
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> list[float]:
        # Heavy computation happens here, isolated from domain
        return self.model.encode(text).tolist()
```

### 4.4. Calculation Tools (`pandas` / `numpy`)
Agents should not "hallucinate" math. Give them tools. These tools live in **Infrastructure** and are injected into the LangGraph.

```python
# src/infrastructure/tools/data_tools.py
import pandas as pd
import numpy as np
from langchain_core.tools import tool

@tool
def calculate_statistics(csv_path: str) -> dict:
    """Calculates mean and std dev for a CSV file."""
    try:
        df = pd.read_csv(csv_path)
        result = {
            "mean": df.mean(numeric_only=True).to_dict(),
            "std_dev": np.std(df.select_dtypes(include=[np.number]), axis=0).to_dict()
        }
        return result
    except Exception as e:
        return {"error": str(e)}
```

---

## 5. Workflows with LangGraph

The workflow remains the same, but now nodes can leverage these specialized tools.

```python
# src/infrastructure/graphs/nodes.py
from src.domain.workflows.state import AgentState
from src.infrastructure.tools.data_tools import calculate_statistics

def analysis_node(state: AgentState):
    # The LLM decides to call the tool
    # ... LangChain tool binding logic ...
    pass
```

---

## 6. Naming Conventions (Additions)

*   **ML Components:** Suffix with `Model` or `Service`.
    *   `SentimentAnalysisModel`, `LocalEmbeddingService`.
*   **Tools:** Suffix with `Tool`.
    *   `DataCleaningTool`, `FinancialCalcTool`.
*   **Settings:** `Settings` class, variables in `UPPER_CASE`.

---

## 7. What NOT To Do (Strict Rules)

1.  **No Heavy ML in Domain:**
    *   ❌ **Don't:** Import `torch`, `transformers`, or `pandas` in `src/domain`.
    *   ✅ **Do:** Define an interface (e.g., `IDataProcessor`) in Domain, and implement it using Pandas in Infrastructure.

2.  **No Global State for Models:**
    *   ❌ **Don't:** Load a Transformer model as a global variable at the top of a file.
    *   ✅ **Do:** Load it inside a class and manage its lifecycle via the Dependency Injection container (Singleton scope).

3.  **Don't Return DataFrames to API:**
    *   ❌ **Don't:** Return a raw `pandas.DataFrame` from a FastAPI endpoint.
    *   ✅ **Do:** Convert it to a JSON-serializable format (List of Dicts) or a Pydantic model before the Presentation layer.

4.  **Don't Hardcode Retry Logic:**
    *   ❌ **Don't:** Write `while` loops with `try/except` manually for API calls.
    *   ✅ **Do:** Use `tenacity` decorators for clean, declarative retry logic.

5.  **Don't Mix Async and Sync Blocking:**
    *   ❌ **Don't:** Run heavy CPU tasks (like `pandas` calculations or `transformer` inference) directly inside an `async def` FastAPI route. This blocks the event loop.
    *   ✅ **Do:** Run heavy synchronous calculations in a separate thread using `fastapi.concurrency.run_in_threadpool` or a background worker (Celery/arq).

---

## 8. Summary Checklist

- [ ] **Pydantic:** Are all API inputs/outputs and Domain entities defined as Pydantic models?
- [ ] **Settings:** Are secrets managed via `pydantic-settings`?
- [ ] **Resilience:** Is `tenacity` applied to external LLM calls?
- [ ] **Isolation:** Are `pandas`/`numpy`/`torch` imports restricted to the `infrastructure` folder?
- [ ] **Performance:** Are heavy ML/Math operations offloaded from the main async event loop?