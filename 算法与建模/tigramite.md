# Tigramite：多变量时间序列因果发现与分析指南

[Tigramite](https://github.com/jakobrunge/tigramite) 是一个用于高维时间序列因果发现（Causal Discovery）与因果推断的 Python 框架。其核心基于 **PCMCI 算法**（由 Jakob Runge 等人提出），能够从多变量时间序列中有效克服自相关、滞后混杂与非线性干扰，精确重构变量间的时序因果网络。

---

## 1. PCMCI 核心原理与优势

传统格兰杰因果（Granger Causality）在高维时序中容易出现**假阳性（False Positives）**与**维数灾难**。PCMCI 分为两个核心阶段：
1. **PC 阶段 (条件筛选)**：利用迭代条件独立性检验，快速为每个变量筛选出最相关的强滞后父节点集合 $\hat{\mathcal{P}}(X_t)$。
2. **MCI 阶段 (瞬间条件独立性检验)**：检验 $Y_{t-\tau} \to X_t$ 时，将 $X_t$ 与 $Y_{t-\tau}$ 的父节点集合全部作为条件进行控制：
   $$\text{MCI}: \quad Y_{t-\tau} \perp X_t \mid \hat{\mathcal{P}}(X_t) \setminus \{Y_{t-\tau}\}, \hat{\mathcal{P}}(Y_{t-\tau})$$
   从而彻底消除假相关与虚假滞后依赖。

---

## 2. 端到端实战代码（含模拟时序数据生成）

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

# 1. 模拟生成带有明确因果关系的三维时序系统 (T=1000)
# 系统设定:
# X0(t) = 0.7 * X0(t-1) + noise
# X1(t) = 0.8 * X1(t-1) + 0.5 * X0(t-1) + noise (X0 滞后1阶因果影响 X1)
# X2(t) = 0.6 * X2(t-1) - 0.4 * X1(t-2) + noise (X1 滞后2阶因果影响 X2)

np.random.seed(42)
T = 1000
data = np.zeros((T, 3))
for t in range(2, T):
    data[t, 0] = 0.7 * data[t-1, 0] + np.random.normal(0, 1)
    data[t, 1] = 0.8 * data[t-1, 1] + 0.5 * data[t-1, 0] + np.random.normal(0, 1)
    data[t, 2] = 0.6 * data[t-1, 2] - 0.4 * data[t-2, 1] + np.random.normal(0, 1)

var_names = ['Sensor_X0', 'Sensor_X1', 'Sensor_X2']

# 2. 包装为 Tigramite 标准 DataFrame
dataframe = pp.DataFrame(data=data, datatime={0: np.arange(T)}, var_names=var_names)

# 3. 初始化条件独立性检验器 (线性选用 ParCorr，非线性可选 GPDC / CMIknn)
parcorr = ParCorr(significance='analytic')
pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=1)

# 4. 运行 PCMCI 因果发现
# tau_max: 最大因果滞后阶数 (比如最多看过去 3 个时间步)
# pc_alpha: PC 阶段条件筛选显著性水平
# alpha_level: MCI 阶段显著性判定阈值
results = pcmci.run_pcmci(tau_max=3, pc_alpha=0.05, alpha_level=0.01)

# 5. 输出因果检验结果矩阵 (p-value 矩阵与因果强度 val_matrix)
print("=== 因果强度矩阵 (val_matrix) ===")
print(results['val_matrix'].round(3))

print("=== 因果显著性判定图 (graph) ===")
print(results['graph'])

# 6. 可视化因果图 (Summary Graph 与 Time Series Graph)
# Summary Graph: 变量间聚合因果拓扑
tp.plot_graph(
    val_matrix=results['val_matrix'],
    graph=results['graph'],
    var_names=var_names,
    link_colorbar_label='MCI Partial Correlation',
    node_colorbar_label='Auto-MCI',
    show_autodependency_lags=False
)
plt.title("Summary Causal Graph")
plt.show()

# Time Series Graph: 展开各个时间步 lag 的因果流动图
tp.plot_time_series_graph(
    figsize=(10, 6),
    val_matrix=results['val_matrix'],
    graph=results['graph'],
    var_names=var_names,
    link_colorbar_label='MCI Partial Correlation'
)
plt.title("Time-Lagged Causal Graph")
plt.show()
```

---

## 3. 核心参数与调优建议

| 参数 | 含义 | 调优建议 |
| :--- | :--- | :--- |
| `tau_max` | 最大滞后时间步数 | 根据物理/业务常识设定（如采样率 1 秒，预期滞后 5 秒内，则设为 5）。设置过大增加计算量。 |
| `cond_ind_test` | 条件独立性检验方法 | • 线性高斯时序：使用 `ParCorr()`（极快）；<br>• 非线性/复杂分布：使用 `GPDC()` (高斯过程距离相关) 或 `CMIknn()` (K近邻互信息)。 |
| `pc_alpha` | 第一阶段变量初筛阈值 | 建议设在 `0.1 ~ 0.2` 之间，略宽一些以保留潜在父节点。 |
| `alpha_level` | 第二阶段最终因果确认显著性 | 严格控制假阳性，一般设为 `0.01` 或 `0.05`。 |
