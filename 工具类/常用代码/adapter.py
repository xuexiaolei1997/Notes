from typing import Dict, Type
import abc

_adapters: Dict[str, Type] = dict()


def register_adapter(adapter_type: str):
    def decorator(cls):
        _adapters[adapter_type] = cls
        return cls
    return decorator


# ↓↓ ↓↓ ↓↓ ↓↓ ↓↓ ↓↓ ↓↓ ↓↓ ↓↓ ↓↓ ↓↓ ↓↓ ↓↓ ↓↓ ↓↓ ↓↓ ↓↓ ↓↓ ↓↓ ↓↓


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                       抽象类，一般不改                    #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class Adapters(abc.ABC):
    def __init__(self) -> None:
        """
        初始化
        """
        pass
    
    @abc.abstractmethod
    def test_func(self):
        """
        测试函数
        """
        pass


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 下方新增适配器，使用 @ 自动注入            #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

@register_adapter("subadapter1")
class SubAdapter1(Adapters):
    def __init__(self) -> None:
        print("子类初始化")
    
    def test_func(self):
        print("子类测试函数")

# ↑↑ ↑↑ ↑↑ ↑↑ ↑↑ ↑↑ ↑↑ ↑↑ ↑↑ ↑↑ ↑↑ ↑↑ ↑↑ ↑↑ ↑↑ ↑↑ ↑↑ ↑↑ ↑↑ ↑↑


