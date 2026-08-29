# pybind11：C++ 与 Python 高效混合编程实战

[pybind11](https://github.com/pybind/pybind11) 是一个轻量级、仅头文件（Header-only）的 C++11 库，用于在 C++ 和 Python 之间暴露类型与接口，性能极高且语法极为优雅，是算法底层加速与跨语言绑定的首选工具。

---

## 1. 核心导出模式与 C++ 代码编写

### 1.1 基础函数与面向对象类绑定 (`example.cpp`)
```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>       // 支持 std::vector, std::map 与 Python list/dict 自动转换
#include <pybind11/numpy.h>     // 支持与 NumPy 数组高效交互

namespace py = pybind11;

// 1. 普通纯 C++ 高性能计算函数
double fast_add(double a, double b) {
    return a + b;
}

// 2. 面向对象 C++ 类
class MatrixCalculator {
public:
    MatrixCalculator(const std::string &name) : name_(name) {}

    void setName(const std::string &name) { name_ = name; }
    const std::string &getName() const { return name_; }

    // 3. NumPy 数组高效处理 (零拷贝/内存视图)
    py::array_t<double> scale_array(py::array_t<double> input_array, double factor) {
        py::buffer_info buf = input_array.request();
        
        // 创建同维度的输出 NumPy 数组
        auto result = py::array_t<double>(buf.size);
        py::buffer_info res_buf = result.request();

        double *ptr_in = static_cast<double *>(buf.ptr);
        double *ptr_out = static_cast<double *>(res_buf.ptr);

        // 密集循环向量化加速
        #pragma omp parallel for
        for (ssize_t i = 0; i < buf.size; i++) {
            ptr_out[i] = ptr_in[i] * factor;
        }

        result.resize(buf.shape);
        return result;
    }

private:
    std::string name_;
};

// 4. 模块导出入口宏
PYBIND11_MODULE(fast_ops, m) {
    m.doc() = "Fast operations implemented in modern C++ with pybind11";

    // 导出独立函数 (可指定默认参数与文档注释)
    m.def("fast_add", &fast_add, "Add two numbers efficiently",
          py::arg("a") = 0.0, py::arg("b") = 0.0);

    // 导出 C++ 类与属性/方法
    py::class_<MatrixCalculator>(m, "MatrixCalculator")
        .def(py::init<const std::string &>(), py::arg("name") = "DefaultCalculator")
        .def_property("name", &MatrixCalculator::getName, &MatrixCalculator::setName)
        .def("scale_array", &MatrixCalculator::scale_array, "Multiply numpy array by a scalar factor",
             py::arg("input_array"), py::arg("factor"));
}
```

---

## 2. 两种构建与编译方式

### 方案 A：使用 `setup.py` (推荐：直接通过 pip 安装分发)

在项目根目录创建 `setup.py`：
```python
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "fast_ops",
        ["example.cpp"],
        extra_compile_args=["-O3", "-std=c++17"],
    ),
]

setup(
    name="fast_ops",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
```
编译与本地开发安装命令：
```bash
# 安装 pybind11 依赖
pip install pybind11

# 以可编辑模式编译安装到当前 Python 环境
pip install -e .
```

---

### 方案 B：使用 `CMake` 构建动态库 (.so / .pyd)

项目目录结构：
```text
try_pybind11/
├── CMakeLists.txt
├── example.cpp
└── extern/
    └── pybind11/ (git submodule 或 clone)
```

`CMakeLists.txt`：
```cmake
cmake_minimum_required(VERSION 3.15)
project(fast_ops LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加 pybind11 子模块
add_subdirectory(extern/pybind11)

# 生成 Python 模块
pybind11_add_module(fast_ops example.cpp)
```

编译步骤：
```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```
> Linux / macOS 下生成 `fast_ops.cpython-*.so`，Windows 下生成 `fast_ops.cp*.pyd`。

---

## 3. Python 端的调用与验证

```python
import numpy as np
import fast_ops

# 1. 调用独立函数
print("C++ 加法结果:", fast_ops.fast_add(10.5, 20.3))  # 30.8

# 2. 调用 C++ 类与 NumPy 数组计算
calc = fast_ops.MatrixCalculator(name="MyGPUCalc")
print("Calculator 名字:", calc.name)

# 传入大型 NumPy 数组
data = np.linspace(1.0, 10.0, 5, dtype=np.float64)
scaled = calc.scale_array(data, factor=2.0)

print("原始数据:", data)
print("C++ 处理后:", scaled)  # [ 2.  4.25  6.5  8.75 20. ]
```
