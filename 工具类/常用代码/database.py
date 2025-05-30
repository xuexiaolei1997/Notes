# database.py
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy import String, Integer, select
import asyncio

# 假设使用SQLite数据库的异步版本
# 注意：对于生产环境，你可能需要使用 asyncpg (PostgreSQL) 或 aiomysql (MySQL) 等异步驱动
SQLALCHEMY_DATABASE_URL = "sqlite+aiosqlite:///./test_async.db"

# 创建异步数据库引擎
engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL,
    echo=True, # 打印SQL语句，方便调试
)

# 创建异步会话工厂
# expire_on_commit=False 意味着在 commit 后，ORM 对象不会自动变为过期状态，
# 你仍然可以访问它们的属性而不需要重新从数据库加载。
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# 声明基类
Base = declarative_base()

# 定义一个简单的用户模型 (可以使用 Mapped 声明式映射，更现代的写法)
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(length=256), index=True)
    email: Mapped[str] = mapped_column(String(length=256), unique=True, index=True)

    def __repr__(self):
        return f"<User(id={self.id}, name='{self.name}', email='{self.email}')>"

# 创建表 (注意这里需要异步执行)
async def create_db_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# 异步依赖注入函数，用于获取数据库会话
async def get_db_async():
    db = AsyncSessionLocal()
    try:
        yield db
    finally:
        await db.close()

# 示例：如何手动创建表 (如果不是在应用程序启动时自动创建)
async def _main():
    await create_db_tables()

if __name__ == "__main__":
    asyncio.run(_main())


