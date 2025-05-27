"""
Perfect Magic Square Genetic Algorithm Solver
Computational Biology - Exercise 2

Perfect Magic Squares (Most Perfect Magic Squares) are defined only for n = 4k (multiples of 4)
and have additional constraints beyond standard magic squares.
"""

import numpy as np
import random
import time
from typing import List, Tuple, Dict, Any
from standard_magic_square import StandardMagicSquare

class PerfectMagicSquare(StandardMagicSquare):
    def __init__(self, n: int, population_size: int = 100, max_generations: int = 2000, 
                 mutation_rate: float = 0.15):
        """
        Initialize Perfect Magic Square solver
        
        Args:
            n: Size of the magic square (must be multiple of 4)
            population_size: Number of individuals in population
            max_generations: Maximum number of generations
            mutation_rate: Probability of mutation
        """
        if n % 4 != 0:
            raise ValueError("Perfect magic squares are only defined for n = 4k (multiples of 4)")
        
        super().__init__(n, population_size, max_generations, mutation_rate)
        
        # Additional constraints for perfect magic squares
        self.sub_square_size = 2  # Size of sub-squares to check
        
    def fitness(self, individual: np.ndarray) -> float:
        """
        Calculate fitness for perfect magic square
        Includes all standard constraints plus perfect magic square constraints
        
        Args:
            individual: n x n numpy array representing magic square
            
        Returns:
            Fitness score (sum of absolute deviations from all constraints)
        """
        self.function_evaluations += 1
        
        # Start with standard magic square fitness
        total_error = super().fitness(individual)
        
        # Additional constraints for perfect magic squares
        
        # 1. Check all 2x2 sub-squares sum to magic_sum
        sub_square_target = self.magic_sum
        for i in range(0, self.n - 1, 2):
            for j in range(0, self.n - 1, 2):
                sub_square = individual[i:i+2, j:j+2]
                sub_sum = np.sum(sub_square)
                total_error += abs(sub_sum - sub_square_target)
        
        # 2. Check complementary pairs constraint
        # Numbers at symmetric positions should sum to n²+1
        complement_sum = self.n * self.n + 1
        
        # Check all pairs that are equidistant from center
        center = (self.n - 1) / 2
        for i in range(self.n):
            for j in range(self.n):
                # Find symmetric position
                sym_i = self.n - 1 - i
                sym_j = self.n - 1 - j
                
                # Only check each pair once
                if i <= sym_i and j <= sym_j and (i != sym_i or j != sym_j):
                    pair_sum = individual[i, j] + individual[sym_i, sym_j]
                    total_error += abs(pair_sum - complement_sum)
        
        # 3. Check bent diagonals (for 4x4 and larger)
        if self.n >= 4:
            # There are multiple bent diagonals to check
            # For simplicity, we'll check a few key ones
            
            # Example bent diagonal patterns for 4x4:
            if self.n == 4:
                bent_diagonals = [
                    [(0,1), (1,0), (2,3), (3,2)],  # One bent diagonal
                    [(0,2), (1,3), (2,0), (3,1)],  # Another bent diagonal
                ]
                
                for diagonal in bent_diagonals:
                    diag_sum = sum(individual[i, j] for i, j in diagonal)
                    total_error += abs(diag_sum - self.magic_sum)
        
        # 4. Additional pandiagonal properties (all parallel diagonals sum to magic sum)
        # Check broken diagonals
        for k in range(1, self.n):
            # Main diagonal direction
            diag_sum1 = 0
            diag_sum2 = 0
            for i in range(self.n):
                j1 = (i + k) % self.n
                j2 = (i - k) % self.n
                diag_sum1 += individual[i, j1]
                diag_sum2 += individual[i, j2]
            
            total_error += abs(diag_sum1 - self.magic_sum)
            total_error += abs(diag_sum2 - self.magic_sum)
            
            # Anti-diagonal direction
            diag_sum3 = 0
            diag_sum4 = 0
            for i in range(self.n):
                j3 = (self.n - 1 - i + k) % self.n
                j4 = (self.n - 1 - i - k) % self.n
                diag_sum3 += individual[i, j3]
                diag_sum4 += individual[i, j4]
            
            total_error += abs(diag_sum3 - self.magic_sum)
            total_error += abs(diag_sum4 - self.magic_sum)
        
        return total_error
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhanced crossover for perfect magic squares
        Uses multiple crossover strategies to maintain structure and unique numbers.
        """
        n = self.n
        size = n * n
        method = random.choice(['order', 'block', 'ring'])

        if method == 'order':
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
                        # FIX: Check bounds before accessing p2_filtered
                        if p2_idx < len(p2_filtered):
                            offspring[i] = p2_filtered[p2_idx]
                            p2_idx += 1
                        else:
                            # Fallback: find any missing number
                            all_nums = set(range(1, n*n + 1))
                            used_nums = set([x for x in offspring if x != -1])
                            missing = list(all_nums - used_nums)
                            if missing:
                                offspring[i] = missing[0]
                return np.array(offspring).reshape(n, n)

            return order_crossover(p1_flat, p2_flat), order_crossover(p2_flat, p1_flat)

        elif method == 'block':
                offspring1_flat = np.full(size, -1)
                offspring2_flat = np.full(size, -1)
                block_size = min(random.randint(1, n // 2), n)
                if block_size > 0:
                    start_i = random.randint(0, max(0, n - block_size))
                    start_j = random.randint(0, max(0, n - block_size))
                    indices = [i * n + j for i in range(start_i, min(start_i + block_size, n))
                            for j in range(start_j, min(start_j + block_size, n))]

                    # Offspring 1: Take block from parent 1
                    block1 = parent1[start_i:start_i + block_size, start_j:start_j + block_size].flatten()
                    for k, idx in enumerate(indices):
                        offspring1_flat[idx] = block1[k]
                    # Fill the rest from parent 2
                    p2_flat = parent2.flatten()
                    p2_idx = 0
                    for i in range(size):
                        if offspring1_flat[i] == -1:
                            while p2_idx < size and p2_flat[p2_idx] in offspring1_flat:
                                p2_idx += 1
                            if p2_idx < size:
                                offspring1_flat[i] = p2_flat[p2_idx]
                                p2_idx += 1

                    # Offspring 2: Take block from parent 2
                    block2 = parent2[start_i:start_i + block_size, start_j:start_j + block_size].flatten()
                    for k, idx in enumerate(indices):
                        offspring2_flat[idx] = block2[k]
                    # Fill the rest from parent 1
                    p1_flat = parent1.flatten()
                    p1_idx = 0
                    for i in range(size):
                        if offspring2_flat[i] == -1:
                            while p1_idx < size and p1_flat[p1_idx] in offspring2_flat:
                                p1_idx += 1
                            if p1_idx < size:
                                offspring2_flat[i] = p1_flat[p1_idx]
                                p1_idx += 1

                    return offspring1_flat.reshape(n, n), offspring2_flat.reshape(n, n)
                else:
                    # Fallback to order crossover if block size is invalid
                    p1_flat = parent1.flatten()
                    p2_flat = parent2.flatten()
                    return self._order_crossover(p1_flat, p2_flat, n, size), self._order_crossover(p2_flat, p1_flat, n, size)

        else:  # ring method
            offspring1_flat = np.full(size, -1)
            offspring2_flat = np.full(size, -1)
            max_ring = n // 2
            if max_ring > 0:
                ring = random.randint(0, max_ring - 1)
                ring_indices = self._get_ring_indices(n, ring)

                # Offspring 1: Take ring from parent 1
                p1_flat = parent1.flatten()
                ring1_values = [p1_flat[i] for i in ring_indices if i < size]
                for k, idx in enumerate(ring_indices):
                    if idx < size and k < len(ring1_values):
                        offspring1_flat[idx] = ring1_values[k]
                # Fill the rest from parent 2
                p2_flat = parent2.flatten()
                p2_idx = 0
                for i in range(size):
                    if offspring1_flat[i] == -1:
                        while p2_idx < size and p2_flat[p2_idx] in offspring1_flat:
                            p2_idx += 1
                        if p2_idx < size:
                            offspring1_flat[i] = p2_flat[p2_idx]
                            p2_idx += 1

                # Offspring 2: Take ring from parent 2
                p2_flat_orig = parent2.flatten()
                ring2_values = [p2_flat_orig[i] for i in ring_indices if i < size]
                for k, idx in enumerate(ring_indices):
                    if idx < size and k < len(ring2_values):
                        offspring2_flat[idx] = ring2_values[k]
                # Fill the rest from parent 1
                p1_flat_orig = parent1.flatten()
                p1_idx = 0
                for i in range(size):
                    if offspring2_flat[i] == -1:
                        while p1_idx < size and p1_flat_orig[p1_idx] in offspring2_flat:
                            p1_idx += 1
                        if p1_idx < size:
                            offspring2_flat[i] = p1_flat_orig[p1_idx]
                            p1_idx += 1

                return offspring1_flat.reshape(n, n), offspring2_flat.reshape(n, n)
            else:
                # Fallback to order crossover if max_ring is invalid
                p1_flat = parent1.flatten()
                p2_flat = parent2.flatten()
                return self._order_crossover(p1_flat, p2_flat, n, size), self._order_crossover(p2_flat, p1_flat, n, size)

    def _order_crossover(self, p1, p2, n, size):
            start, end = sorted(random.sample(range(size), 2))
            offspring = [-1] * size
            offspring[start:end] = p1[start:end]
            p2_filtered = [x for x in p2 if x not in offspring[start:end]]
            p2_idx = 0
            for i in range(size):
                if offspring[i] == -1:
                    offspring[i] = p2_filtered[p2_idx]
                    p2_idx += 1
            return np.array(offspring).reshape(n, n)
        
    def _get_ring_indices(self, nn, rr):
        indices = []
        for j in range(rr, nn - rr): indices.append(rr * nn + j)
        for i in range(rr + 1, nn - rr - 1): indices.append(i * nn + (nn - rr - 1))
        if nn - rr - 1 > rr:
            for j in range(nn - rr - 1, rr - 1, -1): indices.append((nn - rr - 1) * nn + j)
        if rr < nn - rr - 1:
            for i in range(nn - rr - 2, rr, -1): indices.append(i * nn + rr)
        return indices
        
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        Enhanced mutation for perfect magic squares
        Uses structure-preserving mutations. Mutation should not introduce duplicate numbers.
        """
        if random.random() > self.mutation_rate:
            return individual.copy()

        mutant = individual.copy()
        mutation_type = random.choice(['swap', 'rotate_block', 'shift_row_col'])

        if mutation_type == 'swap':
            pos1 = (random.randint(0, self.n-1), random.randint(0, self.n-1))
            pos2 = (random.randint(0, self.n-1), random.randint(0, self.n-1))
            mutant[pos1], mutant[pos2] = mutant[pos2], mutant[pos1]

        elif mutation_type == 'rotate_block':
            if self.n >= 4:
                block_size = 2
                # FIX: Ensure valid start positions
                max_start_i = max(0, self.n - block_size)
                max_start_j = max(0, self.n - block_size)
                if max_start_i >= 0 and max_start_j >= 0:
                    start_i = random.randint(0, max_start_i)
                    start_j = random.randint(0, max_start_j)
                    block = mutant[start_i:start_i+block_size, start_j:start_j+block_size]
                    rotated_block = np.rot90(block, k=random.randint(1, 3))
                    mutant[start_i:start_i+block_size, start_j:start_j+block_size] = rotated_block

        elif mutation_type == 'shift_row_col':
            if random.choice([True, False]):  # Row
                row_idx = random.randint(0, self.n-1)
                shift = random.randint(1, self.n-1)
                mutant[row_idx, :] = np.roll(mutant[row_idx, :], shift)
            else:  # Column
                col_idx = random.randint(0, self.n-1)
                shift = random.randint(1, self.n-1)
                mutant[:, col_idx] = np.roll(mutant[:, col_idx], shift)

        return mutant
    
    def ring_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Crossover based on concentric rings"""
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        # Choose a ring to exchange
        max_ring = self.n // 2
        if max_ring > 0:  # FIX: Ensure valid ring
            ring = random.randint(0, max_ring - 1)
            
            # Get positions in the ring
            positions = []
            
            # Top row
            for j in range(ring, self.n - ring):
                positions.append((ring, j))
            
            # Right column (excluding corners)
            for i in range(ring + 1, self.n - ring - 1):
                positions.append((i, self.n - ring - 1))
            
            # Bottom row (if not the same as top)
            if self.n - ring - 1 > ring:
                for j in range(self.n - ring - 1, ring - 1, -1):
                    positions.append((self.n - ring - 1, j))
            
            # Left column (excluding corners)
            if ring < self.n - ring - 1:
                for i in range(self.n - ring - 2, ring, -1):
                    positions.append((i, ring))
            
            # Exchange values at these positions
            for pos in positions:
                i, j = pos
                offspring1[i, j], offspring2[i, j] = parent2[i, j], parent1[i, j]
        
        return offspring1, offspring2
    
    def local_optimization(self, individual: np.ndarray, steps: int = None) -> np.ndarray:
        """
        Enhanced local optimization for perfect magic squares
        Focuses on fixing the most violated constraints
        """
        if steps is None:
            steps = self.n * 2  # More steps for complex constraints
        
        current = individual.copy()
        current_fitness = self.fitness(current)
        
        for _ in range(steps):
            # Try different types of local moves
            move_type = random.choice(['swap', 'sub_square_fix', 'diagonal_fix'])
            
            if move_type == 'swap':
                # Standard swap
                pos1 = (random.randint(0, self.n-1), random.randint(0, self.n-1))
                pos2 = (random.randint(0, self.n-1), random.randint(0, self.n-1))
                
                neighbor = current.copy()
                neighbor[pos1], neighbor[pos2] = neighbor[pos2], neighbor[pos1]
            
            elif move_type == 'sub_square_fix':
                # Try to fix a 2x2 sub-square
                # FIX: Ensure valid range
                max_i = max(0, self.n - 2)
                max_j = max(0, self.n - 2)
                if max_i >= 0 and max_j >= 0:
                    i = random.randint(0, max_i) if max_i > 0 else 0
                    j = random.randint(0, max_j) if max_j > 0 else 0
                    
                    neighbor = current.copy()
                    
                    # Try swapping within the sub-square
                    positions = [(i, j), (i, min(j+1, self.n-1)), (min(i+1, self.n-1), j), (min(i+1, self.n-1), min(j+1, self.n-1))]
                    valid_positions = [(x, y) for x, y in positions if x < self.n and y < self.n]
                    if len(valid_positions) >= 2:
                        pos1, pos2 = random.sample(valid_positions, 2)
                        neighbor[pos1], neighbor[pos2] = neighbor[pos2], neighbor[pos1]
                    else:
                        continue
                else:
                    continue
            
            else:  # diagonal_fix
                # Try to improve a diagonal
                neighbor = current.copy()
                
                # Choose random diagonal element and swap with another
                diag_pos = random.randint(0, self.n-1)
                other_pos = (random.randint(0, self.n-1), random.randint(0, self.n-1))
                
                neighbor[diag_pos, diag_pos], neighbor[other_pos] = \
                    neighbor[other_pos], neighbor[diag_pos, diag_pos]
            
            neighbor_fitness = self.fitness(neighbor)
            
            # Accept if better (hill climbing)
            if neighbor_fitness < current_fitness:
                current = neighbor
                current_fitness = neighbor_fitness
        
        return current
    
    def is_perfect_magic_square(self, square: np.ndarray) -> bool:
        """Check if the square is a valid perfect magic square"""
        return self.fitness(square) == 0
    
    def print_square(self, square: np.ndarray) -> None:
        """Print perfect magic square with additional information"""
        if square is None:
            print("No solution found")
            return
            
        print(f"\nPerfect Magic Square {self.n}x{self.n}:")
        print("-" * (4 * self.n + 1))
        for row in square:
            print("|" + "|".join(f"{num:3d}" for num in row) + "|")
        print("-" * (4 * self.n + 1))
        print(f"Magic Sum: {self.magic_sum}")
        
        # Check various properties
        is_standard = super().fitness(square) == 0
        is_perfect = self.is_perfect_magic_square(square)
        
        print(f"Standard Magic Square: {is_standard}")
        print(f"Perfect Magic Square: {is_perfect}")
        
        if is_standard:
            # Check 2x2 sub-squares
            print("\n2x2 Sub-square sums:")
            for i in range(0, self.n-1, 2):
                for j in range(0, self.n-1, 2):
                    sub_sum = np.sum(square[i:i+2, j:j+2])
                    print(f"Position ({i},{j}): {sub_sum}", end="  ")
                print()

# Example usage and testing
if __name__ == "__main__":
    print("Testing Perfect Magic Square Solver")
    print("=" * 40)
    
    # Test with 4x4 perfect magic square
    solver = PerfectMagicSquare(n=4, population_size=100, max_generations=1000)
    
    algorithms = ["classic", "darwinian", "lamarckian"]
    
    for algo in algorithms:
        print(f"\nRunning {algo.title()} Algorithm for 4x4 Perfect Magic Square:")
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