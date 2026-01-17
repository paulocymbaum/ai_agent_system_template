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
