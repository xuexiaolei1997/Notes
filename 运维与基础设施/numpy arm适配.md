# 解决 NumPy 在x86与arm机器上的计算速度差异过大问题

## 1 背景

当使用numpy计算大矩阵求逆问题时，x86与arm上运行同一段代码，时间差异巨大，大约在25倍。

|架构|x86|arm|
|-|-|-|
|操作系统|Ubuntu 22.04|kylin 10（2023）|
|cpu num|Intel 16|Phytium 16|
|python|3.10.4|3.10.4|
|numpy|1.26.2|1.26.2|
|scipy|1.15.3|1.15.3|

## 2 问题分析

代码相同的情况下，在不同机器上运行效率不同，问题定位在cpu架构上，在此方向入手进行分析。

首先运行代码

```python
import numpy as np

np.show_config()
```

查看numpy所绑定的cpu的运算库信息，可以看到最直观的差异在 **SIMD** 上：

- x86 (SSE、AVX-512): 每个时钟周期可以处理 512 位数据（8 个双精度浮点数）。

- ARM (NEON/SVE): NEON 通常是 128 位（2 个双精度）。

通过资料查询， NEON 在 arm 机器上的优化不如x86自带的 MKL 计算库，因此定位问题在此处。

同时观察上述代码输出，关键点观察 blas_mkl_info 是否存在。x86 使用 MKL 而 arm 使用未优化的 OpenBLAS，这计算效率差距的直接来源。

另一方面，标准的 `pip install numpy` 在麒麟、鲲鹏等机器上性能不佳，因为其并未引入openblas。需要手动安装openblas，然后从源码编译安装numpy并指定openblas进行优化。

## 3 问题解决

原始代码为:

```python
a = np.asmatrix(x)
b = a.I
```

大矩阵求逆的时间复杂度在 $O(n^3)$，且 `numpy.matrix` 类型在 NumPy 官方文档中已被标记为“不推荐使用（Pending Deprecation）”。`matrix` 对象的 `.I` 操作符在底层执行的是通用的矩阵求逆，且由于其特殊的子类实现，往往无法触发表层代码的最优化路径。针对 x86 与 ARM 的性能差异，分别从数据结构、算法替代、内存连续性三个维度进行优化。

### 算法层面优化

在数值计算中，直接求逆（Inverse）通常是效率最低且数值最不稳定的方案。在数学上，$A^T \cdot D^{-1}$ 等价于求解线性方程组 $D \cdot X = A$ 的转置（即 $X^T$）。

因为计算目的是为了计算 $x^{-1} \cdot v$ 或类似操作，应改用 solve：

```python
b = np.linalg.solve(x, np.eye(x.shape[0])) # 得到逆矩阵 (LU 分解)
```

`np.linalg.inv` 内部调用的是 LAPACK 的 dgetrf (LU 分解) 和 dgetri (求逆)。在 ARM 架构上，由于 Python 封装开销，直接调用 `scipy.linalg` 通常能获得更好的性能，因为 SciPy 对 BLAS/LAPACK 的链接通常比 NumPy 更紧密。在此基础上进一步优化得到：

```python
scipy.linalg.solve(D, x, overwrite_a=True, overwrite_b=False, check_finite=False)
```

### 基础计算库优化

在 kylin 操作系统中安装基础计算库：

```shell
yum update
yum install -y openblas-devel lapack-devel gcc-gfortran
```

导出环境变量：

```shell
export CFLAGS="-O3 -march=armv8-a+simd -mtune=generic -ffast-math"
export NPY_BLAS_ORDER=openblas
export NPY_LAPACK_ORDER=openblas
```

### 库优化

#### 踩坑记录

下载 numpy-1.26.4 源码包，解压后进入目录，执行命令进行编译安装：

```shell
cd numpy-1.26.4

# NumPy 1.26.x 强制要求 meson-python 和 ninja
python3 -m pip install --upgrade pip
python3 -m pip install build meson-python ninja cython

# -march=native: 自动适配当前 ARM 核心的所有指令集 (NEON, LSE, SVE等)
export CFLAGS="-O3 -march=native -mtune=native"
export CXXFLAGS="-O3 -march=native -mtune=native"
export FFLAGS="-O3 -march=native"

# 强制编写 site.cfg (解决链接不到 OpenBLAS 的问题)
# 这一步如果不做，NumPy 会假装找到了 OpenBLAS 实际上没链接上
printf "[openblas]\nlibraries = openblas\nlibrary_dirs = /usr/lib64\ninclude_dirs = /usr/include\nruntime_library_dirs = /usr/lib64\n" > site.cfg

python3 setup.py bdist_wheel

python3 -m pip install dist/*.whl --force-reinstall --no-deps
```

numpy1.26.4版本编译存在 `meson` `meson-python` 的版本不兼容的问题，在解决后还会出现gcc版本过低的问题。kylin 系统默认提供的gcc版本为7.3.0，而numpy1.26.4版本编译需要的gcc/c++版本需要 ≥8.4.0，因此本文采用降级 numpy 进行修改，降级到 numpy 1.25.2。

#### 解决方案

下载 numpy-1.25.2 源码包，解压后进入目录，执行命令进行编译安装：

```shell
#!/bin/bash

# =================================================================
# 针对 Kylin V10 SP1 (ARM64) 的 NumPy 1.25.2 极致优化脚本
# 环境要求: GCC 7.3.0+, OpenBLAS
# =================================================================

set -e

SOURCE_ZIP="numpy-1.25.2.zip"
DIR_NAME="numpy-1.25.2"

# 1. 准备基础依赖
printf "1. 正在安装麒麟系统构建依赖...\n"
sudo yum install -y python3-devel gcc-gfortran openblas-devel lapack-devel unzip

# 2. 安装构建工具 (1.25.x 不需要 Meson，这会简单很多)
printf "2. 安装 Python 构建工具...\n"
pip3 install --upgrade pip
pip3 install "cython<3.0" "setuptools<60.0" wheel  # 1.25.x 建议使用经典工具链

# 3. 解压源码
if [ -d "$DIR_NAME" ]; then rm -rf "$DIR_NAME"; fi
unzip "$SOURCE_ZIP"
cd "$DIR_NAME"

# 4. 配置针对 $7400$ 矩阵的性能参数
printf "4. 注入极致优化参数 (针对 NEON/ARMv8)...\n"
# GCC 7.3 稳定支持的最高优化级别
export CFLAGS="-O3 -march=armv8-a+simd -mtune=generic -fPIC"
export LDFLAGS="-L/usr/lib64"
export OMP_NUM_THREADS=16  # 根据你的核心数调整
export OPENBLAS_NUM_THREADS=16
# 禁用 NumPy 内部较慢的额外并行
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# 5. 写入 site.cfg (指向麒麟系统的 OpenBLAS)
printf "5. 配置底层数学库路径...\n"
printf "[openblas]\n" > site.cfg
printf "libraries = openblas\n" >> site.cfg
printf "library_dirs = /usr/lib64\n" >> site.cfg
printf "include_dirs = /usr/include\n" >> site.cfg
printf "runtime_library_dirs = /usr/lib64\n" >> site.cfg

# 6. 开始编译成 Wheel (直接使用 setup.py)
printf "6. 开始编译 (针对 7400 规模矩阵优化)...\n"
# NPY_NUM_BUILD_JOBS 可以加速编译过程
export NPY_NUM_BUILD_JOBS=$(nproc)
python3 setup.py bdist_wheel

# 7. 输出结果
printf "------------------------------------------------\n"
printf "构建完成！\n"
printf "------------------------------------------------\n"
```

然后进入解压后的numpy目录下，进入dist目录，查看生成的whl文件并且进行手动安装运行：

> python3 -m pip install numpy-1.25.2-cp310-cp310-linux_aarch64.whl

安装完成后运行示例代码，查看运行时间，提升5倍。
