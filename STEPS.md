# Project Setup Steps: AI Agent System

This document provides step-by-step instructions to build the complete project skeleton for the AI Agent Backend system using Groq as the LLM provider.

---

## Prerequisites

- Python 3.11 or higher
- Docker & Docker Compose (for PostgreSQL and Redis)
- Git
- Virtual environment tool (venv or poetry)

---

## Step 1: Initialize Project Structure

### 1.1 Create Root Directory and Initialize Git

```bash
mkdir ai_agent_system
cd ai_agent_system
git init
```

### 1.2 Create Complete Directory Structure

```bash
# Create all necessary directories
mkdir -p src/{config,domain/{agents,workflows},application/{dtos,use_cases},infrastructure/{db/repositories,llm,ml,tools,graphs}}
mkdir -p tests/{unit,integration}
mkdir -p scripts
mkdir -p data/{embeddings,cache}
mkdir -p logs
```

The final structure should look like:

```
ai_agent_system/
├── .env.example
├── .env
├── .gitignore
├── README.md
├── ARCHITECTURE.md
├── STEPS.md
├── requirements.txt
├── docker-compose.yml
├── pyproject.toml
│
├── src/
│   ├── __init__.py
│   ├── main.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   │
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── agents/
│   │   │   ├── __init__.py
│   │   │   ├── entities.py
│   │   │   └── ports.py
│   │   └── workflows/
│   │       ├── __init__.py
│   │       └── state.py
│   │
│   ├── application/
│   │   ├── __init__.py
│   │   ├── dtos/
│   │   │   ├── __init__.py
│   │   │   ├── request.py
│   │   │   └── response.py
│   │   └── use_cases/
│   │       ├── __init__.py
│   │       └── run_analysis.py
│   │
│   └── infrastructure/
│       ├── __init__.py
│       ├── db/
│       │   ├── __init__.py
│       │   ├── models.py
│       │   └── repositories/
│       │       ├── __init__.py
│       │       └── agent_repository.py
│       │
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── client.py
│       │   └── resilience.py
│       │
│       ├── ml/
│       │   ├── __init__.py
│       │   └── embeddings.py
│       │
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── data_tools.py
│       │   └── web_tools.py
│       │
│       └── graphs/
│           ├── __init__.py
│           ├── nodes.py
│           └── workflow.py
│
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── __init__.py
│   │   └── test_use_cases.py
│   └── integration/
│       ├── __init__.py
│       └── test_api.py
│
├── scripts/
│   ├── init_db.py
│   └── seed_data.py
│
├── data/
│   ├── embeddings/
│   └── cache/
│
└── logs/
```

---

## Step 2: Create Configuration Files

### 2.1 Create `requirements.txt`

```bash
cat > requirements.txt << 'EOF'
# Core Framework
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-multipart==0.0.6

# Async HTTP Client
httpx==0.26.0

# Data Validation & Settings
pydantic==2.5.3
pydantic-settings==2.1.0

# LLM Orchestration
langchain==0.1.4
langchain-groq==0.0.1
langchain-community==0.0.13
langchain-core==0.1.15
langgraph==0.0.20

# Resilience & Retries
tenacity==8.2.3

# Database
sqlalchemy==2.0.25
psycopg2-binary==2.9.9
alembic==1.13.1

# Redis Cache
redis==5.0.1

# Local ML & Embeddings
sentence-transformers==2.3.1
transformers==4.36.2
torch==2.1.2

# Data Processing & Analysis
pandas==2.2.0
numpy==1.26.3

# Utilities
python-dotenv==1.0.0
python-json-logger==2.0.7

# Development & Testing
pytest==7.4.4
pytest-asyncio==0.23.3
pytest-cov==4.1.0
black==24.1.1
ruff==0.1.14
mypy==1.8.0
EOF
```

### 2.2 Create `.env.example`

```bash
cat > .env.example << 'EOF'
# Application Settings
APP_NAME=AI Agent System
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# API Keys - LLM Provider (Groq)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=mixtral-8x7b-32768

# Database Configuration
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ai_agent_db
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=10

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_MAX_CONNECTIONS=10

# Local ML Configuration
EMBEDDING_MODEL_PATH=all-MiniLM-L6-v2
ML_DEVICE=cpu
CACHE_EMBEDDINGS=true

# LangGraph Configuration
MAX_ITERATIONS=10
TIMEOUT_SECONDS=300

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
CORS_ORIGINS=["http://localhost:3000"]

# Resilience Configuration
MAX_RETRIES=3
RETRY_MIN_WAIT=4
RETRY_MAX_WAIT=10

# File Paths
DATA_DIR=./data
LOGS_DIR=./logs
CACHE_DIR=./data/cache
EOF
```

### 2.3 Create `.gitignore`

```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/
.venv

# Environment Variables
.env
.env.local

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Logs
logs/
*.log

# Database
*.db
*.sqlite
*.sqlite3

# ML Models Cache
data/embeddings/
data/cache/
models/

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# MyPy
.mypy_cache/
.dmypy.json
dmypy.json

# Jupyter
.ipynb_checkpoints

# OS
.DS_Store
Thumbs.db
EOF
```

### 2.4 Create `docker-compose.yml`

```bash
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  postgres:
    image: postgres:16-alpine
    container_name: ai_agent_postgres
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: ai_agent_db
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: ai_agent_redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5

volumes:
  postgres_data:
  redis_data:
EOF
```

### 2.5 Create `pyproject.toml` (Optional - for modern Python tooling)

```bash
cat > pyproject.toml << 'EOF'
[tool.black]
line-length = 100
target-version = ['py311']
include = '\.pyi?$'

[tool.ruff]
line-length = 100
target-version = "py311"
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
plugins = ["pydantic.mypy"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "-v --cov=src --cov-report=html --cov-report=term"
EOF
```

---

## Step 3: Implement Core Files

### 3.1 Configuration Layer

**File: `src/config/settings.py`**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    APP_NAME: str = "AI Agent System"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # Groq API
    GROQ_API_KEY: str
    GROQ_MODEL: str = "mixtral-8x7b-32768"
    
    # Database
    DATABASE_URL: str
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_MAX_CONNECTIONS: int = 10
    
    # Local ML
    EMBEDDING_MODEL_PATH: str = "all-MiniLM-L6-v2"
    ML_DEVICE: str = "cpu"
    CACHE_EMBEDDINGS: bool = True
    
    # LangGraph
    MAX_ITERATIONS: int = 10
    TIMEOUT_SECONDS: int = 300
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    
    # Resilience
    MAX_RETRIES: int = 3
    RETRY_MIN_WAIT: int = 4
    RETRY_MAX_WAIT: int = 10
    
    # Paths
    DATA_DIR: str = "./data"
    LOGS_DIR: str = "./logs"
    CACHE_DIR: str = "./data/cache"
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)


settings = Settings()
```

### 3.2 Domain Layer

**File: `src/domain/agents/entities.py`**

```python
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
```

**File: `src/domain/agents/ports.py`**

```python
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
```

**File: `src/domain/workflows/state.py`**

```python
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
```

### 3.3 Application Layer

**File: `src/application/dtos/request.py`**

```python
from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    """Request DTO for running an analysis."""
    
    query: str = Field(..., min_length=1, description="The analysis query")
    context: dict | None = Field(default=None, description="Additional context")
    max_iterations: int | None = Field(default=None, ge=1, le=20)
```

**File: `src/application/dtos/response.py`**

```python
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
```

**File: `src/application/use_cases/run_analysis.py`**

```python
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
            max_iterations=request.max_iterations or settings.MAX_ITERATIONS
        )
        
        return AnalysisResponse(
            task_id=task_id,
            status="completed",
            result=result.get("final_result"),
            iterations=len(result.get("intermediate_steps", [])),
            started_at=started_at,
            completed_at=datetime.utcnow()
        )
```

### 3.4 Infrastructure Layer

**File: `src/infrastructure/llm/client.py`**

```python
from langchain_groq import ChatGroq
from src.config.settings import settings


def create_llm_client() -> ChatGroq:
    """Factory function to create a Groq LLM client."""
    
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model_name=settings.GROQ_MODEL,
        temperature=0.7,
        max_tokens=2048
    )
```

**File: `src/infrastructure/llm/resilience.py`**

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from src.config.settings import settings


def create_retry_decorator():
    """Create a retry decorator for LLM calls."""
    
    return retry(
        stop=stop_after_attempt(settings.MAX_RETRIES),
        wait=wait_exponential(
            multiplier=1,
            min=settings.RETRY_MIN_WAIT,
            max=settings.RETRY_MAX_WAIT
        ),
        retry=retry_if_exception_type((Exception,)),
        reraise=True
    )


# Decorator instance
robust_llm_call = create_retry_decorator()
```

**File: `src/infrastructure/ml/embeddings.py`**

```python
from sentence_transformers import SentenceTransformer
from src.domain.agents.ports import EmbeddingService
from src.config.settings import settings


class LocalHuggingFaceEmbeddings(EmbeddingService):
    """Local embedding service using HuggingFace models."""
    
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_PATH
        self.model = SentenceTransformer(self.model_name)
    
    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query."""
        return self.model.encode(text).tolist()
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple documents."""
        embeddings = self.model.encode(texts)
        return [emb.tolist() for emb in embeddings]
```

**File: `src/infrastructure/tools/data_tools.py`**

```python
import pandas as pd
import numpy as np
from langchain_core.tools import tool


@tool
def calculate_statistics(data: list[float]) -> dict:
    """Calculate statistical measures for a list of numbers.
    
    Args:
        data: List of numeric values
        
    Returns:
        Dictionary with mean, median, std_dev, min, max
    """
    try:
        arr = np.array(data)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std_dev": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": len(data)
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def analyze_csv(csv_path: str) -> dict:
    """Analyze a CSV file and return basic statistics.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Dictionary with column statistics
    """
    try:
        df = pd.read_csv(csv_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        result = {
            "rows": len(df),
            "columns": list(df.columns),
            "numeric_columns": numeric_cols,
            "statistics": df[numeric_cols].describe().to_dict() if numeric_cols else {}
        }
        return result
    except Exception as e:
        return {"error": str(e)}
```

**File: `src/infrastructure/tools/web_tools.py`**

```python
import httpx
from langchain_core.tools import tool


@tool
async def fetch_url(url: str) -> dict:
    """Fetch content from a URL.
    
    Args:
        url: The URL to fetch
        
    Returns:
        Dictionary with status_code and content
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            return {
                "status_code": response.status_code,
                "content": response.text[:1000],  # Limit content
                "headers": dict(response.headers)
            }
    except Exception as e:
        return {"error": str(e)}
```

**File: `src/infrastructure/graphs/nodes.py`**

```python
from src.domain.workflows.state import AgentState
from langchain_core.messages import HumanMessage, AIMessage


def router_node(state: AgentState) -> AgentState:
    """Route the request to appropriate agent."""
    
    # Simple routing logic
    state["current_agent"] = "analysis_agent"
    return state


def analysis_node(state: AgentState) -> AgentState:
    """Perform analysis using LLM and tools."""
    
    # Placeholder - actual implementation would use the LLM
    messages = state["messages"]
    
    # Add a mock response
    state["messages"] = [
        *messages,
        AIMessage(content=f"Analysis completed for: {state['task_description']}")
    ]
    
    state["intermediate_steps"].append({
        "agent": "analysis_agent",
        "action": "analysis_complete"
    })
    
    return state


def finalize_node(state: AgentState) -> AgentState:
    """Finalize the workflow and produce final result."""
    
    state["final_result"] = f"Analysis completed with {len(state['intermediate_steps'])} steps"
    return state
```

**File: `src/infrastructure/graphs/workflow.py`**

```python
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
```

**File: `src/infrastructure/db/models.py`**

```python
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
import uuid

Base = declarative_base()


class AgentModel(Base):
    """SQLAlchemy model for agents."""
    
    __tablename__ = "agents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    role = Column(String(255), nullable=False)
    model = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TaskModel(Base):
    """SQLAlchemy model for tasks."""
    
    __tablename__ = "tasks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), nullable=False)
    description = Column(Text, nullable=False)
    status = Column(String(50), default="pending")
    result = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
```

### 3.5 Presentation Layer

**File: `src/main.py`**

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from src.config.settings import settings
from src.application.dtos.request import AnalysisRequest
from src.application.dtos.response import AnalysisResponse
from src.application.use_cases.run_analysis import RunAnalysisUseCase
from src.infrastructure.graphs.workflow import create_workflow
from src.infrastructure.llm.client import create_llm_client

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    logger.info("Starting AI Agent System...")
    app.state.workflow = create_workflow()
    app.state.llm_client = create_llm_client()
    yield
    # Shutdown
    logger.info("Shutting down AI Agent System...")


app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": settings.APP_NAME,
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    """Run an analysis using the agent workflow."""
    try:
        # Create a simple workflow executor
        class WorkflowExecutor:
            def __init__(self, workflow):
                self.workflow = workflow
            
            async def run(self, query: str, context: dict, max_iterations: int):
                from langchain_core.messages import HumanMessage
                
                initial_state = {
                    "messages": [HumanMessage(content=query)],
                    "current_agent": "",
                    "task_description": query,
                    "intermediate_steps": [],
                    "final_result": None
                }
                
                result = await self.workflow.ainvoke(initial_state)
                return result
        
        executor = WorkflowExecutor(app.state.workflow)
        use_case = RunAnalysisUseCase(executor)
        result = await use_case.execute(request)
        return result
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
```

---

## Step 4: Setup and Installation

### 4.1 Create Python Virtual Environment

```bash
# Using venv
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using poetry (alternative)
poetry init
poetry shell
```

### 4.2 Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4.3 Setup Environment Variables

```bash
# Copy example to actual .env file
cp .env.example .env

# Edit .env and add your Groq API key
nano .env  # or vim, code, etc.
```

**Important:** Add your actual Groq API key to `.env`:
```
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```

### 4.4 Start Infrastructure Services

```bash
# Start PostgreSQL and Redis
docker-compose up -d

# Verify services are running
docker-compose ps
```

### 4.5 Initialize Database

Create a simple initialization script:

**File: `scripts/init_db.py`**

```python
from sqlalchemy import create_engine
from src.infrastructure.db.models import Base
from src.config.settings import settings

def init_database():
    """Initialize database tables."""
    engine = create_engine(settings.DATABASE_URL)
    Base.metadata.create_all(engine)
    print("Database initialized successfully!")

if __name__ == "__main__":
    init_database()
```

Run the initialization:

```bash
python scripts/init_db.py
```

---

## Step 5: Run and Test

### 5.1 Start the Application

```bash
# Development mode with auto-reload
python src/main.py

# Or using uvicorn directly
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 5.2 Test the API

**Test with curl:**

```bash
# Health check
curl http://localhost:8000/health

# Run analysis
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Analyze the current market trends",
    "context": {"industry": "technology"},
    "max_iterations": 5
  }'
```

**Test with Python:**

```python
import httpx
import asyncio

async def test_analysis():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/analyze",
            json={
                "query": "What are the key features of AI agents?",
                "max_iterations": 3
            }
        )
        print(response.json())

asyncio.run(test_analysis())
```

### 5.3 Run Tests

Create a basic test file:

**File: `tests/integration/test_api.py`**

```python
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "service" in response.json()


def test_health():
    """Test health check."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_analyze():
    """Test analysis endpoint."""
    response = client.post(
        "/api/v1/analyze",
        json={
            "query": "Test query",
            "max_iterations": 3
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert "status" in data
```

Run tests:

```bash
pytest tests/ -v
```

---

## Step 6: Code Quality and Linting

### 6.1 Format Code with Black

```bash
black src/ tests/
```

### 6.2 Lint with Ruff

```bash
ruff check src/ tests/
```

### 6.3 Type Checking with MyPy

```bash
mypy src/
```

---

## Step 7: Project Documentation

### 7.1 Update README.md

Add project overview, installation instructions, and usage examples.

### 7.2 Generate API Documentation

FastAPI automatically generates API docs:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Step 8: Deployment Preparation

### 8.1 Create Production Requirements

```bash
pip freeze > requirements-lock.txt
```

### 8.2 Create Docker Image (Optional)

**File: `Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY .env .env

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t ai-agent-system .
docker run -p 8000:8000 --env-file .env ai-agent-system
```

---

## Step 9: Verification Checklist

- [ ] All directories created with proper `__init__.py` files
- [ ] `requirements.txt` includes all necessary dependencies
- [ ] `.env` file configured with valid Groq API key
- [ ] PostgreSQL and Redis running via docker-compose
- [ ] Database tables initialized
- [ ] Application starts without errors
- [ ] Health endpoint responds successfully
- [ ] Analysis endpoint accepts requests
- [ ] Tests pass successfully
- [ ] Code formatted with Black
- [ ] No linting errors from Ruff
- [ ] Type checking passes with MyPy

---

## Step 10: Next Steps

1. **Implement Full LangGraph Workflow:** Expand nodes with actual LLM interactions
2. **Add Authentication:** Implement JWT or API key authentication
3. **Add More Tools:** Expand the tools directory with domain-specific tools
4. **Implement Caching:** Use Redis for LLM response caching
5. **Add Monitoring:** Integrate observability tools (Prometheus, Grafana)
6. **Create Migration Scripts:** Use Alembic for database migrations
7. **Add Background Tasks:** Implement Celery or ARQ for async processing
8. **Implement RAG:** Add vector database (Pinecone, Qdrant, or pgvector)
9. **Add Streaming:** Implement SSE for streaming responses
10. **Production Deploy:** Deploy to cloud platform (AWS, GCP, Azure)

---

## Document Matrix

| Document | Purpose | Notes |
| --- | --- | --- |
| [README.md](README.md) | Entry point for developers and operators with overview, getting-started steps, and usage patterns. | Mirrors Step 7 guidance and links to API docs. |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Captures the high-level system components, module boundaries, and deployment considerations. | Keeps evolving as LangGraph workflow and infrastructure grow. |
| [STEPS.md](STEPS.md) | This playbook describing setup, configuration, and validation steps. | Source of truth for onboarding and repeated environments. |
| [requirements.txt](requirements.txt) | Lists production and development Python dependencies required for the service stack. | Referenced by `pip install` and Docker builds. |
| [docker-compose.yml](docker-compose.yml) | Defines PostgreSQL and Redis services for local development. | Ensure alignment with `.env` settings and health checks. |
| [pyproject.toml](pyproject.toml) | Centralizes tooling config (Black, Ruff, MyPy, Pytest) for format, lint, type, and test commands. | Update whenever tooling versions change. |
| [.env.example](.env.example) | Template for environment variables consumed by `src/config/settings.py`. | Copy to `.env` and populate secrets before running. |
| [.gitignore](.gitignore) | Prevents build artifacts, caches, and credentials from entering version control. | Keep in sync with new directories (e.g., data, logs). |
| [scripts/init_db.py](scripts/init_db.py) | Simple script to bootstrap database tables via SQLAlchemy metadata. | Run after spinning up PostgreSQL. |

---

## Troubleshooting

### Common Issues

**Issue: Module not found errors**
```bash
# Ensure you're in the virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

**Issue: Database connection failed**
```bash
# Check if PostgreSQL is running
docker-compose ps

# Restart services
docker-compose restart postgres
```

**Issue: Groq API errors**
- Verify API key is correct in `.env`
- Check Groq API quota and rate limits
- Ensure `langchain-groq` is installed correctly

**Issue: PyTorch installation fails**
```bash
# Use CPU-only version for smaller installations
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Resources

- **Groq Documentation:** https://console.groq.com/docs
- **LangChain Documentation:** https://python.langchain.com/
- **LangGraph Documentation:** https://langchain-ai.github.io/langgraph/
- **FastAPI Documentation:** https://fastapi.tiangolo.com/
- **Pydantic Documentation:** https://docs.pydantic.dev/

---

**Project Setup Complete!** 🎉

You now have a fully functional AI Agent System skeleton ready for development.
