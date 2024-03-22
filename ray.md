Ray 是一个任务级别分配的分布式框架, Ray的系统层是以Task为抽象粒度的，用户可以在代码里任意生成和组合task，比如拆分成多个Stage,每个Task执行什么逻辑，每个task需要多少资源,非常自由，对资源把控力很强

## 无状态

```python
import ray

ray.init()


@ray.remote
def square(x):
    return x ** 2


features = [square.remote(i) for i in range(400)]
print(ray.get(features))
```

## 有状态

```python
import ray

ray.init()


@ray.remote
class Counter:
    def __init__(self):
        self.i = 0

def get(self):
        return self.i

    def increase(self, value):
        self.i += value


c = Counter.remote()
for _ in range(100):
    c.increase.remote(10)

print(ray.get(c.get.remote()))
```

pailler加密
