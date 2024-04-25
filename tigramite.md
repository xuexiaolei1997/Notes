# TIGRAMITE

TIGRAMITE 是一个时间序列数据分析的python包，它基于PCMCI框架，可以从离散或连续值的时间序列中重建图形模型（条件独立性图），并创建高质量的结果图。

导包：

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(action='ignore')

import tigramite
import tigramite.data_processing as pp
import tigramite.plotting as tp


from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
```

初始化数据：

```python
var_names = list(data.columns)
dataframe = pp.DataFrame(data=data.to_numpy(), datatime={0: np.arange(len(data))}, var_names=var_names)
```

绘图：

```python
tp.plot_timeseries(dataframe=dataframe)  # 绘制时间序列

matrix_lags = None
tp.plot_scatterplots(dataframe, add_scatterplot_args={'matrix_lags': matrix_lags})  # 绘制散点图
tp.plot_densityplots(dataframe, add_densityplot_args={'matrix_lags': matrix_lags})  # 绘制密度函数图
```

初始化因果关系所需类：

```python
parcorr = ParCorr()
pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)
```

计算滞后性：

```python
lagged_dependencies = pcmci.get_lagged_dependencies(tau_max=2, val_only=True)['val_matrix']
tp.plot_lagfuncs(lagged_dependencies, setup_args={'var_names': var_names})
plt.show()
```

运行：

```python
results = pcmci.run_pcmci(tau_max=2, pc_alpha=0.1, alpha_level=0.01)

tp.plot_graph(graph=results['graph'], val_matrix=results['val_matrix'], var_names=var_names, show_autodependency_lags=False)
plt.show()

tp.plot_time_series_graph(figsize=(10, 8), val_matrix=results['val_matrix'], graph=results['graph'], var_names=var_names)
plt.show()
```
