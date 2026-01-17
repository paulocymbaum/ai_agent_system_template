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
