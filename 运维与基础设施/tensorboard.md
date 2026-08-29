# TensorBoard 训练监控与可视化指南

---

## 1. PyTorch 中标准记录流程

在模型训练过程中，通过 `torch.utils.tensorboard.SummaryWriter` 记录标量（Loss/Metric）、图像、模型计算图与参数分布直方图：

```python
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# 1. 初始化 Writer，建议按实验命名子目录 (如 runs/exp_transformer_lr0.001)
writer = SummaryWriter(log_dir="runs/exp_lstm_v1")

# 2. 模拟训练循环记录 Loss 与准确率
for epoch in range(100):
    train_loss = 1.0 / (epoch + 1)
    val_loss = 1.2 / (epoch + 1)
    
    # 记录单个标量
    writer.add_scalar("Loss/Train", train_loss, global_step=epoch)
    writer.add_scalar("Loss/Val", val_loss, global_step=epoch)
    
    # 记录多个标量在同一张图表对比
    writer.add_scalars("Loss/Comparison", {"train": train_loss, "val": val_loss}, global_step=epoch)

# 3. 记录模型网络结构计算图 (Graph)
dummy_input = torch.randn(1, 10)
model = nn.Linear(10, 2)
writer.add_graph(model, dummy_input)

# 4. 训练结束关闭 Writer
writer.flush()
writer.close()
```

---

## 2. 启动服务与常用参数

```bash
# 本地启动 TensorBoard (指定日志上级目录)
tensorboard --logdir=runs --port=6006

# 允许局域网/公网所有 IP 访问
tensorboard --logdir=runs --host=0.0.0.0 --port=6006

# 限制每类数据的采样数量 (防止超大日志导致前端卡死)
tensorboard --logdir=runs --samples_per_plugin scalars=500,images=50
```

---

## 3. 远程服务器可视化 (SSH 端口转发)

当在没有图形界面的远程 Linux 服务器或云主机上训练时，可通过本地终端建立 SSH 隧道：

```bash
# 在本地电脑终端执行：将远程服务器的 6006 端口映射到本地的 6006
ssh -L 6006:localhost:6006 username@remote_server_ip -p 22

# 然后在本地浏览器直接打开：
# http://localhost:6006
```

---

## 4. 常见排错与避坑指南

### 4.1 界面打开后完全空白 (Blank Page)
**原因**：通常是 `torch-tb-profiler` 插件版本与当前 TensorBoard 前端资源冲突。  
**解决办法**：
```bash
pip uninstall -y torch-tb-profiler
# 重启 tensorboard 即可恢复正常显示
tensorboard --logdir=runs --port=6006
```

### 4.2 训练曲线迟迟不更新
**原因**：TensorBoard 默认会将数据缓存在内存中批量刷盘。  
**解决办法**：
- 在关键 step 或 epoch 结束处显式调用 `writer.flush()`。
- 启动时设置更短的重新加载周期：`tensorboard --logdir=runs --reload_interval=10`。
