import random

class Particle:
    def __init__(self, position, velocity, target_func):
        self.position = position
        self.velocity = velocity
        self.best_position = position[:]
        self.target_func = target_func
        self.best_value = float('inf')
    
    def calculate_value(self, position):
        x = position[0]
        return self.target_func(x)
    
    def update_velocity(self, global_best_position, w, c1, c2):
        for i in range(len(self.position)):
            r1 = random.random()
            r2 = random.random()
            cognitive_velocity = c1 * r1 * (self.best_position[i] - self.position[i])
            social_velocity = c2 * r2 * (global_best_position[i] - self.position[i])
            self.velocity[i] = w * self.velocity[i] + cognitive_velocity + social_velocity
    
    def update_position(self, bounds):
        for i in range(len(self.position)):
            self.position[i] += self.velocity[i]
            if self.position[i] < bounds[i][0]:
                self.position[i] = bounds[i][0]
            elif self.position[i] > bounds[i][1]:
                self.position[i] = bounds[i][1]
        
        current_value = self.calculate_value(self.position)
        if current_value < self.best_value:
            self.best_position = self.position[:]
            self.best_value = current_value

class PSO:
    def __init__(self, num_particles, dimensions, bounds, w, c1, c2, iterations, target_func):
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.bounds = bounds
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.iterations = iterations
        self.global_best_position = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dimensions)]
        self.global_best_value = float('inf')
        self.target_func = target_func
    
    def initialize_particles(self):
        self.particles = []
        for _ in range(self.num_particles):
            position = [random.uniform(self.bounds[i][0], self.bounds[i][1]) for i in range(self.dimensions)]
            velocity = [random.uniform(-1, 1) for _ in range(self.dimensions)]
            particle = Particle(position, velocity, self.target_func)
            self.particles.append(particle)
    
    def run(self):
        self.initialize_particles()
        for iteration in range(self.iterations):
            for particle in self.particles:
                particle.update_velocity(self.global_best_position, self.w, self.c1, self.c2)
                particle.update_position(self.bounds)
                if particle.best_value < self.global_best_value:
                    self.global_best_position = particle.best_position[:]
                    self.global_best_value = particle.best_value
            
            print(f"Iteration {iteration}: Best value = {self.global_best_value}, Best position = {self.global_best_position[0]}")

# 参数定义
NUM_PARTICLES = 30  # 粒子数量
DIMENSIONS = 1  # 维度
BOUNDS = [(-10, 10)]  # 搜索空间的边界
W = 0.5  # 惯性权重
C1 = 1.5  # 认知系数
C2 = 1.5  # 社会系数
ITERATIONS = 50  # 迭代次数

TARGET_FUNC = lambda x: x ** 2 + x * 3 + 2  # 优化函数

# 运行粒子群优化
pso = PSO(NUM_PARTICLES, DIMENSIONS, BOUNDS, W, C1, C2, ITERATIONS, TARGET_FUNC)
pso.run()
