from collections.abc import AsyncGenerator

from sqlalchemy.pool import NullPool
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from api.config import Settings


def get_async_database_url() -> str:
    if Settings.DATABASE_URL.startswith("postgresql+asyncpg://"):
        return Settings.DATABASE_URL
    if Settings.DATABASE_URL.startswith("postgresql://"):
        return Settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
    return Settings.DATABASE_URL


engine = create_async_engine(
    get_async_database_url(),
    poolclass=NullPool,
)
async_session = async_sessionmaker(engine, expire_on_commit=False)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session
