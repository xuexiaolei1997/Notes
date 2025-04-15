from enum import Enum

class BackendExceptionType(Enum):
    CUSTOM_ERROR: int = 999


class CustomException(Exception):
    """自定义异常"""
    def __init__(self, message, error_code) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
    
    def __str__(self):
        return f"异常码：{self.error_code}, 异常信息：{self.message}"
