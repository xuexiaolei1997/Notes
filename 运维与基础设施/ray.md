# Ray 分布式计算与任务调度指南

Ray 是专为 AI 和大规模 Python 计算设计的分布式计算框架。Ray 系统层以 **Task (无状态函数)** 和 **Actor (有状态类)** 为核心抽象粒度，支持极细粒度的资源配置（CPU/GPU/内存），兼具高性能与灵活性。

---

## 1. 核心概念与基础使用

### 1.1 无状态任务 (Tasks)
适用于纯函数、无内部状态的数据并行处理：

```python
import ray

# 初始化本地 Ray 集群 (可指定 CPU/GPU 资源与内存)
ray.init(num_cpus=4, ignore_reinit_error=True)

# 定义远程函数 (Task)
@ray.remote(num_cpus=1)
def square(x: int) -> int:
    return x ** 2

# 异步并发执行 400 个 Task
futures = [square.remote(i) for i in range(400)]

# 同步获取结果
results = ray.get(futures)
print(f"计算完成，前 10 个结果: {results[:10]}")
```

### 1.2 有状态实体 (Actors)
适用于维护上下文状态的服务、模型推理实例、分布式计数器或多方安全计算节点：

```python
import ray

@ray.remote(num_cpus=1)
class Counter:
    def __init__(self):
        self.value = 0

    def get(self) -> int:
        return self.value

    def increase(self, delta: int = 1) -> int:
        self.value += delta
        return self.value

# 实例化 Actor
counter_actor = Counter.remote()

# 并发异步调用 Actor 方法
futures = [counter_actor.increase.remote(10) for _ in range(100)]
ray.get(futures)

# 获取最终状态
final_val = ray.get(counter_actor.get.remote())
print(f"Counter 最终值: {final_val}")  # 1000
```

---

## 2. 进阶场景：分布式安全计算 (Paillier 同态加密)

在联邦学习与分布式隐私计算中，可利用 Ray 的 Actor 分别模拟客户端（数据方）与服务端（聚合方），配合 `phe` (Python Paillier) 实现密文状态下的安全聚合：

```python
import ray
from phe import paillier

@ray.remote
class DataClient:
    def __init__(self, public_key):
        self.pubkey = public_key

    def encrypt_gradient(self, value: float):
        # 客户端使用公钥对梯度/数据进行同态加密
        return self.pubkey.encrypt(value)

# 服务端生成密钥对
public_key, private_key = paillier.generate_paillier_keypair()

# 创建多个分布式客户端 Actor
clients = [DataClient.remote(public_key) for _ in range(5)]

# 并行在各节点对敏感数据进行同态加密
encrypted_data_refs = [client.encrypt_gradient.remote(float(i * 1.5)) for i, client in enumerate(clients)]
encrypted_data = ray.get(encrypted_data_refs)

# 服务端在不解密的情况下直接进行密文同态加法
encrypted_sum = sum(encrypted_data)

# 服务端私钥解密得到聚合结果
decrypted_sum = private_key.decrypt(encrypted_sum)
print(f"密文同态聚合解密结果: {decrypted_sum}")
```

---

## 3. Ray 集群运维与部署

### 启动 Head 节点与接入 Worker
```bash
# 1. 在主服务器启动 Head 节点 (暴露 Dashboard 和 Redis/GCS 端口)
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265

# 2. 在工作节点机器接入集群
ray start --address='<head_node_ip>:6379' --num-cpus=16 --num-gpus=2

# 3. 查看集群状态与资源节点
ray status

# 4. 停止当前节点的 Ray 进程
ray stop
```

### 生产常用调优参数
- **Object Store 内存上限**：`ray.init(object_store_memory=10 * 1024 * 1024 * 1024)`（限制 Plasma 共享内存为 10GB）
- **Dashboard 可视化界面**：访问 `http://<head_node_ip>:8265` 查看所有 Node、Actor、Task 及日志运行监控。
