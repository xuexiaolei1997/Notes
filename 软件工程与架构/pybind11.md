# python调用C代码

---

## 1、项目准备

新建一个文件夹try_pybind11，用于存放项目文件

在文件夹中新建一个名为"extern"的文件夹

进入文件夹，执行命令  `git clone https://github.com/pybind/pybind11.git` ，下载pybind源码

---

## 2、编写C代码

```c
#include <pybind11/pybind11.h>

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(try_pybind11, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");  // add function
}
```

## 3、Cmake

创建CMakeList.txt

```cmake
cmake_minimum_required(VERSION 3.23)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(try_pybind11)

add_subdirectory(extern/pybind11)

find_package(Python 3.8 COMPONENTS Interpreter Development)

pybind11_add_module(try_pybind11 try_pybind11.cpp)
```

新建一个文件夹"build"，然后进入build文件夹，执行命令

`cmake ..`

`make`

注：windows下，由于没有make，需要安装Visual Studio，执行命令为

`cmake ..`

`cmake --build .`

即可生成对应操作下同下的文件。Linux下为so文件，windows下为pyd文件

---

## 4、python调用

新建一个python文件，调用生成的包，即可直接使用函数

```python
>>> import try_pybind11
>>> try_pybind11.add(10, 20)
>>> 30
```
