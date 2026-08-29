import numpy as np
from typing import Callable, List, Tuple


class Particle:
    """粒子群个体，记录当前位置、速度以及历史个体极值"""
    def __init__(self, position: np.ndarray, velocity: np.ndarray, target_func: Callable[[np.ndarray], float]):
        self.position = np.asarray(position, dtype=np.float64)
        self.velocity = np.asarray(velocity, dtype=np.float64)
        self.best_position = self.position.copy()
        self.target_func = target_func
        self.best_value = float(self.target_func(self.position))
    
    def evaluate(self) -> float:
        """评估当前位置适应度并更新个体历史极值"""
        current_val = float(self.target_func(self.position))
        if current_val < self.best_value:
            self.best_value = current_val
            self.best_position = self.position.copy()
        return current_val

    def update_velocity(
        self,
        global_best_position: np.ndarray,
        w: float,
        c1: float,
        c2: float,
        v_max: np.ndarray
    ):
        """更新粒子飞行速度: v(t+1) = w*v(t) + c1*r1*(pbest - x) + c2*r2*(gbest - x)"""
        dim = len(self.position)
        r1 = np.random.uniform(0, 1, size=dim)
        r2 = np.random.uniform(0, 1, size=dim)
        
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive + social
        
        # 速度边界约束截断 (防止粒子飞行速度过快发生震荡发散)
        self.velocity = np.clip(self.velocity, -v_max, v_max)

    def update_position(self, bounds: np.ndarray):
        """更新粒子位置并施加搜索空间边界约束"""
        self.position += self.velocity
        self.position = np.clip(self.position, bounds[:, 0], bounds[:, 1])


class PSO:
    """标准粒子群优化算法 (Particle Swarm Optimization)
    - 支持多维变量与任意连续目标函数
    - 采用动态线性递减惯性权重策略 (w_max -> w_min)
    - 严格的速度边界与位置边界约束
    """
    def __init__(
        self,
        target_func: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        num_particles: int = 50,
        iterations: int = 100,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 2.0,
        c2: float = 2.0
    ):
        self.target_func = target_func
        self.bounds = np.array(bounds, dtype=np.float64)
        self.dim = len(bounds)
        self.num_particles = num_particles
        self.iterations = iterations
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        
        # 最大速度设为搜索范围跨度的 20%
        self.v_max = 0.2 * (self.bounds[:, 1] - self.bounds[:, 0])

        self.particles: List[Particle] = []
        self.global_best_position = np.zeros(self.dim)
        self.global_best_value = float('inf')
        self.history: List[float] = []

    def initialize_particles(self):
        """初始化粒子群位置与初始速度"""
        self.particles = []
        self.global_best_value = float('inf')
        lows = self.bounds[:, 0]
        highs = self.bounds[:, 1]

        for _ in range(self.num_particles):
            pos = np.random.uniform(lows, highs, size=self.dim)
            vel = np.random.uniform(-self.v_max, self.v_max, size=self.dim)
            particle = Particle(pos, vel, self.target_func)
            self.particles.append(particle)

            if particle.best_value < self.global_best_value:
                self.global_best_value = particle.best_value
                self.global_best_position = particle.best_position.copy()

    def run(self) -> Tuple[np.ndarray, float]:
        """运行粒子群算法迭代"""
        self.initialize_particles()

        for it in range(self.iterations):
            # 动态线性递减惯性权重: 前期注重全局探索，后期注重局部精细收敛
            w = self.w_max - (self.w_max - self.w_min) * (it / self.iterations)

            for particle in self.particles:
                particle.update_velocity(self.global_best_position, w, self.c1, self.c2, self.v_max)
                particle.update_position(self.bounds)
                
                val = particle.evaluate()
                if val < self.global_best_value:
                    self.global_best_value = val
                    self.global_best_position = particle.best_position.copy()

            self.history.append(self.global_best_value)
            if it % 20 == 0 or it == self.iterations - 1:
                print(f"Iteration {it:3d} | 最优目标值: {self.global_best_value:.6f} | 最优坐标: {np.round(self.global_best_position, 4)}")

        return self.global_best_position, self.global_best_value


if __name__ == '__main__':
    # 示例目标函数：2 维 Rosenbrock 香蕉函数 (极小值在 [1.0, 1.0] 处，最小值为 0.0)
    # f(x, y) = 100 * (y - x^2)^2 + (1 - x)^2
    def rosenbrock(x: np.ndarray) -> float:
        return 100.0 * (x[1] - x[0]**2)**2 + (1.0 - x[0])**2

    search_bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    
    print("--- 启动多维粒子群优化 (PSO) 寻优 ---")
    pso = PSO(
        target_func=rosenbrock,
        bounds=search_bounds,
        num_particles=40,
        iterations=80,
        w_max=0.9,
        w_min=0.4,
        c1=2.0,
        c2=2.0
    )
    best_pos, best_val = pso.run()
    print(f"\n寻优完成！全局最优解: {best_pos.round(4)}, 最优目标值: {best_val:.6f}")
