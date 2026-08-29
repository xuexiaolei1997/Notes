# Python 工程实践与高级语言特性速查手册

本文档汇总 Python 开发中的核心工程命令、离线部署、编译加速、动态反射与高级函数特性。

---

## 1. 虚拟环境与离线依赖管理

### 1.1 虚拟环境创建与激活
```bash
# 1. 创建虚拟环境 (my_venv)
python3 -m venv my_venv

# 2. 激活虚拟环境 (Linux / macOS)
source my_venv/bin/activate

# 3. 激活虚拟环境 (Windows PowerShell / CMD)
# my_venv\Scripts\Activate.ps1
# my_venv\Scripts\activate.bat

# 4. 退出虚拟环境
deactivate
```

### 1.2 离线环境依赖打包与安装 (局域网/隔离服务器部署)
```bash
# 1. 在联网机器上批量下载依赖包到指定目录
pip download -d ./wheelhouse -r requirements.txt

# 2. 仅下载指定包的 wheel 二进制文件
pip download -d ./wheelhouse --only-binary :all: numpy pandas torch

# 3. 在离线机器上从本地目录安装 (无需网络连接)
pip install -r requirements.txt --no-index --find-links=./wheelhouse
```

### 1.3 常用国内镜像加速源
```bash
# 临时指定清华源安装
pip install package_name -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 永久设置镜像源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
```

---

## 2. 字节码打包与编译加速

```bash
# 1. 编译单个 Python 文件为 .pyc 字节码
python -m py_compile main.py

# 2. 批量编译整个目录下的所有 Python 文件
python -m compileall -b .

# 3. 使用 Cython 将 Python 模块编译为 C/C++ 动态链接库 (.so / .pyd)
cythonize -3 -i my_module.py
```

---

## 3. 面向对象与动态属性反射机制

在算法库与基础框架设计中（如 Scikit-learn 风格的 Estimator），经常需要支持动态关键字参数注入与反射更新：

```python
from typing import Dict, Any

class BaseEstimator:
    def __init__(self, **kwargs):
        self._custom_params: Dict[str, Any] = {}
        self.set_params(**kwargs)

    def set_params(self, **params: Any) -> "BaseEstimator":
        """动态设置并更新对象属性与超参数"""
        for key, value in params.items():
            # 1. 动态挂载到实例属性
            setattr(self, key, value)
            
            # 2. 如果存在私有下划线属性，同步更新
            if hasattr(self, f"_{key}"):
                setattr(self, f"_{key}", value)
                
            # 3. 记录额外配置字典
            self._custom_params[key] = value
        return self

    def get_params(self) -> Dict[str, Any]:
        """获取所有当前设置的参数"""
        return self._custom_params
```

---

## 4. 常用函数装饰器与类型注解

```python
import time
from functools import wraps
from typing import Callable, Any

def timer(func: Callable) -> Callable:
    """统计函数执行耗时的通用装饰器"""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        print(f"[Timer] 函数 '{func.__name__}' 执行耗时: {elapsed:.4f} 秒")
        return result
    return wrapper

@timer
def heavy_matrix_computation(n: int = 1000):
    return [i ** 2 for i in range(n)]
```

---

## 5. 远程 Jupyter Notebook 安全配置

当在没有公网图形界面的远程 GPU 服务器上使用 Jupyter 时：

### 5.1 生成配置与密码 Hash
```bash
# 1. 生成 Jupyter 配置文件
jupyter notebook --generate-config

# 2. 在 Python 中生成密码密文 Hash
python -c "from notebook.auth import passwd; print(passwd('your_password'))"
# 输出格式类似: argon2:$argon2id$... 或 sha1:...
```

### 5.2 修改配置文件 (`~/.jupyter/jupyter_notebook_config.py`)
```python
# 允许任意 IP 访问
c.NotebookApp.ip = '0.0.0.0'

# 填入上一步生成的密码密文
c.NotebookApp.password = u'argon2:$argon2id$...'

# 禁止自动打开浏览器
c.NotebookApp.open_browser = False

# 自定义服务端口
c.NotebookApp.port = 8888

# 允许远程 root 用户运行 (如果在 Docker 容器内)
c.NotebookApp.allow_root = True
```

### 5.3 启动与 SSH 端口转发
```bash
# 后台启动 Jupyter
nohup jupyter notebook > jupyter.log 2>&1 &

# 在本地电脑建立 SSH 隧道直连
ssh -N -L 8888:localhost:8888 username@remote_ip
# 本地浏览器直接打开 http://localhost:8888 即可访问
```
