import random

# 定义个体
class Individual:
    def __init__(self, chromosome, target_function):
        self.chromosome = chromosome
        self.target_func = target_function
        self.fitness = self.calculate_fitness()

    @staticmethod
    def create_gnome(range_low, range_high):
        return random.uniform(range_low, range_high)

    def calculate_fitness(self):
        x = self.chromosome
        return self.target_func(x)

# 定义遗传算法
class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, crossover_rate, generations, target_func):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations

        self.target_func = target_func

    def initialize_population(self):
        self.population = [Individual(Individual.create_gnome(RANGE_LOW, RANGE_HIGH), self.target_func) for _ in range(self.population_size)]

    def select_parent(self):
        return random.choice(self.population)

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            alpha = random.random()
            child_chromosome = alpha * parent1.chromosome + (1 - alpha) * parent2.chromosome
            return Individual(child_chromosome, self.target_func)
        else:
            return parent1

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            individual.chromosome += random.uniform(-1, 1)
        individual.fitness = individual.calculate_fitness()
        return individual

    def run(self):
        self.initialize_population()
        for generation in range(self.generations):
            new_population = []
            for _ in range(self.population_size):
                parent1 = self.select_parent()
                parent2 = self.select_parent()
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            self.population = sorted(new_population, key=lambda x: x.fitness)
            print(f"Generation {generation}: Best fitness = {self.population[0].fitness}, Best chromosome = {self.population[0].chromosome}")

# 参数定义
POPULATION_SIZE = 100  # 种群大小
MUTATION_RATE = 0.01  # 突变率
CROSSOVER_RATE = 0.9  # 交叉率
GENERATIONS = 300  # 迭代代数

TARGET_FUNC = lambda x: x ** 2 + x * 3 + 2  # 优化函数
RANGE_LOW, RANGE_HIGH = -10, 10

# 运行遗传算法
ga = GeneticAlgorithm(POPULATION_SIZE, MUTATION_RATE, CROSSOVER_RATE, GENERATIONS, TARGET_FUNC)
ga.run()
