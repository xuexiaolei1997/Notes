# 记录python中的命令

## python编译文件

`python -m py_compile test.py`

## vscode 终端配置权限

`sudo chown -R username   /path`

## 下载包的源

https://pypi.tuna.tsinghua.edu.cn/simple/ [清华源](https://pypi.tuna.tsinghua.edu.cn/simple/ "清华源")

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

## python编译

`python -m py_compile test.py`  -> 编译单个文件 pyc
`python -m compileall .` -> 编译文件夹下所有文件

`cythonize -i test.py` -> 编译为pyd

## 远程jupyter配置

1、生成配置文件

jupyter notebook --generate-config

2、生成密码

打开ipython，输入

from notebook.auth import passwd

passwd()

运行后会提示输入密码，随便输入

1，这个密码是用来登录jupyter notebook的，然后会生成一个密钥，复制

3、修改配置文件

在 jupyter_notebook_config.py （~/.jupyter/）中找到下面的行，取消注释并修改

c.NotebookApp.ip='*'

c.NotebookApp.password = u'刚才复制的那个密文'

c.NotebookApp.open_browser = False

c.NotebookApp.port =55555#可自行指定一个端口, 访问时使用该端口

## pip仅下载不安装

pip download package_name

pip download package_name==version

pip download --only-binary :all: package_name

pip download -d /dir -r requirements.txt

## pip从whl安装

pip install -r requirements.txt --no-index --find-links=/dir
