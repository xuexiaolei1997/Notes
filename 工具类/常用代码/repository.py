# repository.py
from typing import TypeVar, Generic, Type, List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession # 导入 AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import declarative_base

# 导入上面定义的Base，用于类型提示
Base = declarative_base()

# 定义一个类型变量，用于表示ORM模型
ModelType = TypeVar("ModelType", bound=Base)

class BaseRepository(Generic[ModelType]):
    """
    一个通用的BaseRepository，用于封装基本的异步CRUD操作。
    """

    def __init__(self, model: Type[ModelType], db: AsyncSession):
        """
        初始化BaseRepository。

        Args:
            model (Type[ModelType]): 绑定的SQLAlchemy ORM模型类。
            db (AsyncSession): 异步数据库会话。
        """
        self.model = model
        self.db = db

    async def get(self, obj_id: Any) -> Optional[ModelType]:
        """
        根据主键获取单个实体。

        Args:
            obj_id (Any): 实体的主键值。

        Returns:
            Optional[ModelType]: 找到的实体，如果不存在则返回None。
        """
        # 使用 select 语句和 execute 获取结果，然后使用 .scalar_one_or_none()
        result = await self.db.execute(select(self.model).filter_by(id=obj_id))
        return result.scalar_one_or_none()

    async def get_multi(
        self,
        *,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ModelType]:
        """
        获取多个实体。

        Args:
            skip (int): 跳过的记录数。
            limit (int): 返回的最大记录数。
            filters (Optional[Dict[str, Any]]): 过滤条件，例如 {"name": "test"}。

        Returns:
            List[ModelType]: 实体列表。
        """
        stmt = select(self.model)
        if filters:
            for key, value in filters.items():
                if hasattr(self.model, key):
                    stmt = stmt.filter(getattr(self.model, key) == value)
        
        stmt = stmt.offset(skip).limit(limit)
        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def add(self, obj_in: ModelType) -> ModelType:
        """
        添加单个实体。

        Args:
            obj_in (ModelType): 要添加的实体对象。

        Returns:
            ModelType: 添加后的实体对象（已包含数据库生成的主键等）。
        """
        self.db.add(obj_in)
        await self.db.flush() # 异步 flush
        await self.db.refresh(obj_in) # 异步 refresh
        return obj_in

    async def create(self, obj_in: Dict[str, Any]) -> ModelType:
        """
        从字典创建并添加一个新实体。

        Args:
            obj_in (Dict[str, Any]): 包含实体属性的字典。

        Returns:
            ModelType: 创建并添加后的实体对象。
        """
        db_obj = self.model(**obj_in)
        self.db.add(db_obj)
        await self.db.flush()
        await self.db.refresh(db_obj)
        return db_obj

    async def update(
        self, db_obj: ModelType, obj_in: Dict[str, Any]
    ) -> ModelType:
        """
        更新现有实体。

        Args:
            db_obj (ModelType): 要更新的现有实体对象。
            obj_in (Dict[str, Any]): 包含要更新属性的字典。

        Returns:
            ModelType: 更新后的实体对象。
        """
        for field, value in obj_in.items():
            if hasattr(db_obj, field):
                setattr(db_obj, field, value)
        # 不再需要 self.db.add(db_obj)，因为 db_obj 已经在会话中被追踪了
        await self.db.flush()
        await self.db.refresh(db_obj)
        return db_obj

    async def delete(self, db_obj: ModelType) -> ModelType:
        """
        删除实体。

        Args:
            db_obj (ModelType): 要删除的实体对象。

        Returns:
            ModelType: 被删除的实体对象。
        """
        await self.db.delete(db_obj) # 异步 delete
        await self.db.flush()
        return db_obj

    async def commit(self):
        """
        提交当前会话的事务。
        """
        await self.db.commit()

    async def rollback(self):
        """
        回滚当前会话的事务。
        """
        await self.db.rollback()

    async def refresh(self, db_obj: ModelType):
        """
        刷新实体，从数据库重新加载其属性。
        """
        await self.db.refresh(db_obj)


