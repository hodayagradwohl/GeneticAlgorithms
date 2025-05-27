"""
Standard Magic Square Genetic Algorithm Solver
Computational Biology - Exercise 2
"""

import numpy as np
import random
import time
from typing import List, Tuple, Dict, Any

class StandardMagicSquare:
    def __init__(self, n: int, population_size: int = 100, max_generations: int = 1000, 
                 mutation_rate: float = 0.1):
        """
        Initialize Standard Magic Square solver
        
        Args:
            n: Size of the magic square (n x n)
            population_size: Number of individuals in population
            max_generations: Maximum number of generations
            mutation_rate: Probability of mutation
        """
        self.n = n
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.magic_sum = n * (n * n + 1) // 2  # Target sum for each row/column/diagonal
        self.numbers = list(range(1, n * n + 1))  # Numbers 1 to n^2
        
        # Statistics tracking
        self.function_evaluations = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
    def create_individual(self) -> np.ndarray:
        """Create a random individual (magic square candidate)"""
        numbers = self.numbers.copy()
        random.shuffle(numbers)
        return np.array(numbers).reshape(self.n, self.n)
    
    def create_population(self) -> List[np.ndarray]:
        """Create initial population"""
        return [self.create_individual() for _ in range(self.population_size)]
    
    def fitness(self, individual: np.ndarray) -> float:
        """
        Calculate fitness of an individual
        Lower fitness is better (0 = perfect magic square)
        
        Args:
            individual: n x n numpy array representing magic square
            
        Returns:
            Fitness score (sum of absolute deviations from magic sum)
        """
        self.function_evaluations += 1
        
        total_error = 0
        
        # Check rows
        for i in range(self.n):
            row_sum = np.sum(individual[i, :])
            total_error += abs(row_sum - self.magic_sum)
        
        # Check columns
        for j in range(self.n):
            col_sum = np.sum(individual[:, j])
            total_error += abs(col_sum - self.magic_sum)
        
        # Check main diagonal
        main_diag_sum = np.sum(np.diag(individual))
        total_error += abs(main_diag_sum - self.magic_sum)
        
        # Check anti-diagonal
        anti_diag_sum = np.sum(np.diag(np.fliplr(individual)))
        total_error += abs(anti_diag_sum - self.magic_sum)
        
        return total_error
    
    def tournament_selection(self, population: List[np.ndarray], 
                           fitness_scores: List[float], tournament_size: int = 3) -> np.ndarray:
        """Select individual using tournament selection"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx].copy()
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform crossover between two parents.
        Ensures offspring contain unique numbers from 1 to n*n.
        Using a similar Order Crossover approach.
        """
        size = self.n * self.n
        p1_flat = parent1.flatten()
        p2_flat = parent2.flatten()

        def order_crossover(p1, p2):
            start, end = sorted(random.sample(range(size), 2))
            offspring = [-1] * size
            offspring[start:end] = p1[start:end]
            p2_filtered = [x for x in p2 if x not in offspring[start:end]]
            p2_idx = 0
            for i in range(size):
                if offspring[i] == -1:
                    offspring[i] = p2_filtered[p2_idx]
                    p2_idx += 1
            return np.array(offspring)

        offspring1_flat = order_crossover(p1_flat, p2_flat)
        offspring2_flat = order_crossover(p2_flat, p1_flat)

        return offspring1_flat.reshape(self.n, self.n), offspring2_flat.reshape(self.n, self.n)
    
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        Mutate individual by swapping two random positions
        """
        if random.random() < self.mutation_rate:
            mutant = individual.copy()
            # Swap two random positions
            pos1 = (random.randint(0, self.n-1), random.randint(0, self.n-1))
            pos2 = (random.randint(0, self.n-1), random.randint(0, self.n-1))
            mutant[pos1], mutant[pos2] = mutant[pos2], mutant[pos1]
            return mutant
        return individual.copy()
    
    def local_optimization(self, individual: np.ndarray, steps: int = None) -> np.ndarray:
        """
        Perform local optimization by trying swaps to improve standard magic square properties.
        Focuses on reducing the error in row, column, and diagonal sums.
        """
        if steps is None:
            steps = self.n

        current = individual.copy()
        current_fitness = self.fitness(current)

        for _ in range(steps):
            # 1. Select a row/column/diagonal index with an error
            errors = []
            for i in range(self.n):
                errors.append((abs(np.sum(current[i, :]) - self.magic_sum), 'row', i))
                errors.append((abs(np.sum(current[:, i]) - self.magic_sum), 'col', i))
            errors.append((abs(np.sum(np.diag(current)) - self.magic_sum), 'diag', 0))
            errors.append((abs(np.sum(np.diag(np.fliplr(current))) - self.magic_sum), 'anti_diag', 0))

            errors.sort(key=lambda x: x[0], reverse=True)  # Focus on larger errors

            if not errors or errors[0][0] == 0:
                break  # No significant errors left

            _, error_type, index = errors[0]

            # 2. Select two random cells to swap, potentially focusing on the problematic row/column/diagonal
            r1, c1 = random.randint(0, self.n - 1), random.randint(0, self.n - 1)
            r2, c2 = random.randint(0, self.n - 1), random.randint(0, self.n - 1)

            neighbor = current.copy()
            neighbor[r1, c1], neighbor[r2, c2] = neighbor[r2, c2], neighbor[r1, c1]
            neighbor_fitness = self.fitness(neighbor)

            # 3. Accept the swap only if it improves the fitness
            if neighbor_fitness < current_fitness:
                current = neighbor
                current_fitness = neighbor_fitness

        return current
    
    def run(self, algorithm_type: str = "classic") -> Dict[str, Any]:
        """
        Run the genetic algorithm
        
        Args:
            algorithm_type: "classic", "darwinian", or "lamarckian"
            
        Returns:
            Dictionary with results
        """
        start_time = time.time()
        self.function_evaluations = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
        # Initialize population
        population = self.create_population()
        
        best_individual = None
        best_fitness = float('inf')
        convergence_generation = None
        stagnation_counter = 0
        max_stagnation = 50  # Stop if no improvement for 50 generations
        
        for generation in range(self.max_generations):
            # Evaluate population
            if algorithm_type == "classic":
                fitness_scores = [self.fitness(ind) for ind in population]
            else:
                # For Darwinian and Lamarckian, evaluate after local optimization
                optimized_population = []
                fitness_scores = []
                
                for ind in population:
                    optimized_ind = self.local_optimization(ind)
                    optimized_population.append(optimized_ind)
                    fitness_scores.append(self.fitness(optimized_ind))
                
                if algorithm_type == "lamarckian":
                    # In Lamarckian evolution, use optimized individuals for next generation
                    population = optimized_population
            
            # Track statistics
            current_best_fitness = min(fitness_scores)
            current_avg_fitness = np.mean(fitness_scores)
            
            self.best_fitness_history.append(current_best_fitness)
            self.avg_fitness_history.append(current_avg_fitness)
            
            # Update best solution
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[np.argmin(fitness_scores)].copy()
                stagnation_counter = 0
                
                if best_fitness == 0:  # Perfect solution found
                    convergence_generation = generation
                    break
            else:
                stagnation_counter += 1
            
            # Check for early stopping
            if stagnation_counter >= max_stagnation:
                break
            
            # Create next generation
            new_population = []
            
            # Elitism: keep best individuals
            elite_count = max(1, self.population_size // 10)  # Top 10%
            elite_indices = np.argsort(fitness_scores)[:elite_count]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                
                offspring1, offspring2 = self.crossover(parent1, parent2)
                
                offspring1 = self.mutate(offspring1)
                offspring2 = self.mutate(offspring2)
                
                new_population.extend([offspring1, offspring2])
            
            # Trim to exact population size
            population = new_population[:self.population_size]
        
        end_time = time.time()
        
        return {
            'best_solution': best_individual,
            'best_fitness': best_fitness,
            'final_avg_fitness': current_avg_fitness,
            'generations_run': generation + 1,
            'function_evaluations': self.function_evaluations,
            'execution_time': end_time - start_time,
            'convergence_generation': convergence_generation,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history
        }
    
    def is_valid_magic_square(self, square: np.ndarray) -> bool:
        """Check if the square is a valid magic square"""
        return self.fitness(square) == 0
    
    def print_square(self, square: np.ndarray) -> None:
        """Print magic square in formatted way"""
        if square is None:
            print("No solution found")
            return
            
        print(f"\nMagic Square {self.n}x{self.n}:")
        print("-" * (4 * self.n + 1))
        for row in square:
            print("|" + "|".join(f"{num:3d}" for num in row) + "|")
        print("-" * (4 * self.n + 1))
        print(f"Magic Sum: {self.magic_sum}")
        print(f"Valid: {self.is_valid_magic_square(square)}")

# Example usage and testing
if __name__ == "__main__":
    # Test with 3x3 magic square
    print("Testing Standard Magic Square Solver")
    print("=" * 40)
    
    solver = StandardMagicSquare(n=3, population_size=50, max_generations=500)
    
    # Test all three algorithms
    algorithms = ["classic", "darwinian", "lamarckian"]
    
    for algo in algorithms:
        print(f"\nRunning {algo.title()} Algorithm:")
        result = solver.run(algo)
        
        print(f"Best Fitness: {result['best_fitness']}")
        print(f"Generations: {result['generations_run']}")
        print(f"Function Evaluations: {result['function_evaluations']}")
        print(f"Execution Time: {result['execution_time']:.2f}s")
        
        if result['best_solution'] is not None:
            solver.print_square(result['best_solution'])
        else:
            print("No solution found")
        
        print("-" * 40)