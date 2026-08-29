import random
import numpy as np
from typing import Callable, List, Tuple


class Individual:
    """遗传算法个体类，代表搜索空间中的一个候选解向量"""
    def __init__(self, chromosome: np.ndarray, target_function: Callable[[np.ndarray], float]):
        self.chromosome = np.asarray(chromosome, dtype=np.float64)
        self.target_func = target_function
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self) -> float:
        """计算适应度 (默认求目标函数最小值，适应度越小越优)"""
        return float(self.target_func(self.chromosome))


class GeneticAlgorithm:
    """实数编码通用连续遗传算法 (Real-Coded GA)
    - 支持多维变量寻优与上下界约束
    - 采用锦标赛选择 (Tournament Selection) + 算术交叉 (Arithmetic Crossover)
    - 结合高斯变异与精英保留机制 (Elitism)
    """
    def __init__(
        self,
        target_func: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        population_size: int = 100,
        generations: int = 150,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.15,
        elitism_count: int = 2
    ):
        self.target_func = target_func
        self.bounds = np.array(bounds, dtype=np.float64)
        self.dim = len(bounds)
        self.pop_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        
        self.population: List[Individual] = []
        self.best_history: List[float] = []

    def initialize_population(self):
        """在边界范围内随机均匀初始化种群"""
        self.population = []
        lows = self.bounds[:, 0]
        highs = self.bounds[:, 1]
        for _ in range(self.pop_size):
            chrom = np.random.uniform(lows, highs, size=self.dim)
            self.population.append(Individual(chrom, self.target_func))

    def tournament_selection(self, k: int = 3) -> Individual:
        """锦标赛选择算子：随机抽取 k 个个体，返回其中适应度最佳者"""
        selected = random.sample(self.population, k)
        selected.sort(key=lambda ind: ind.fitness)
        return selected[0]

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[np.ndarray, np.ndarray]:
        """算术交叉 (Arithmetic Crossover)"""
        if random.random() < self.crossover_rate:
            alpha = np.random.uniform(0, 1, size=self.dim)
            child1_chrom = alpha * parent1.chromosome + (1 - alpha) * parent2.chromosome
            child2_chrom = (1 - alpha) * parent1.chromosome + alpha * parent2.chromosome
            return child1_chrom, child2_chrom
        return parent1.chromosome.copy(), parent2.chromosome.copy()

    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """高斯扰动变异并执行边界截断"""
        for i in range(self.dim):
            if random.random() < self.mutation_rate:
                range_span = self.bounds[i, 1] - self.bounds[i, 0]
                noise = np.random.normal(0, 0.1 * range_span)
                chromosome[i] += noise
                chromosome[i] = np.clip(chromosome[i], self.bounds[i, 0], self.bounds[i, 1])
        return chromosome

    def run(self) -> Tuple[np.ndarray, float]:
        """执行遗传算法演化寻优"""
        self.initialize_population()
        
        for gen in range(self.generations):
            # 1. 种群按适应度升序排列（求最小值）
            self.population.sort(key=lambda ind: ind.fitness)
            best_ind = self.population[0]
            self.best_history.append(best_ind.fitness)

            if gen % 25 == 0 or gen == self.generations - 1:
                print(f"Gen {gen:3d} | 最优适应度 (Loss): {best_ind.fitness:.6f} | 最优解: {np.round(best_ind.chromosome, 4)}")

            # 2. 精英保留 (Elitism)
            next_generation = [self.population[i] for i in range(self.elitism_count)]

            # 3. 生成新一代个体
            while len(next_generation) < self.pop_size:
                p1 = self.tournament_selection()
                p2 = self.tournament_selection()
                c1_chrom, c2_chrom = self.crossover(p1, p2)
                
                c1_chrom = self.mutate(c1_chrom)
                next_generation.append(Individual(c1_chrom, self.target_func))
                
                if len(next_generation) < self.pop_size:
                    c2_chrom = self.mutate(c2_chrom)
                    next_generation.append(Individual(c2_chrom, self.target_func))

            self.population = next_generation

        self.population.sort(key=lambda ind: ind.fitness)
        return self.population[0].chromosome, self.population[0].fitness


if __name__ == '__main__':
    # 示例目标函数：2 维经典 Sphere 函数 f(x, y) = (x - 2)^2 + (y + 3)^2 + 5
    # 理论全局最小值在 x = 2.0, y = -3.0 处，最小值为 5.0
    def sphere_target(x: np.ndarray) -> float:
        return (x[0] - 2.0) ** 2 + (x[1] + 3.0) ** 2 + 5.0

    search_bounds = [(-10.0, 10.0), (-10.0, 10.0)]
    
    print("--- 启动多维实数遗传算法 (GA) 寻优 ---")
    ga = GeneticAlgorithm(
        target_func=sphere_target,
        bounds=search_bounds,
        population_size=80,
        generations=100,
        crossover_rate=0.9,
        mutation_rate=0.2
    )
    best_solution, best_val = ga.run()
    print(f"\n寻优完成！全局最优解: {best_solution.round(4)}, 目标函数最小值: {best_val:.6f}")
