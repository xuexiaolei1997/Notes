# 记录python中的命令

## python编译文件

`python -m py_compile test.py`

## vscode 终端配置权限

`sudo chown -R username   /path`

## 下载包的源

https://pypi.tuna.tsinghua.edu.cn/simple/

## python新建虚拟环境

`python -m venv my_venv`

`source my_venv/bin/activate`

## 通过类关键字参数设置变量

```python
from typing import Dict, Any

class C:
    def __init__(self, **kwargs):
        self._other_params: Dict[str, Any] = {}
        self.set_params(**kwargs)
  
    def set_params(self, **params: dict):
        """Set the parameters of this class.

        Parameters
        ----------
        **params
            Parameter names with their new values.

        Returns
        -------
        None
        """
        for key, value in params.items():
            setattr(self, key, value)
            if hasattr(self, f"_{key}"):
                setattr(self, f"_{key}", value)
            self._other_params[key] = value
```
