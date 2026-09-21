"""
CSCI-561 HW1 — 3D TSP via Genetic Algorithm (Python 3.7-safe).

City  = (x, y, z)           # three ints
Tour  = [city, city, ...]   # length N, each city once; return-to-start is
                            # added when scoring / writing output.txt

This file is a readable "see how the pieces fit" version. Study it, then
rewrite the GA parts yourself so the work is yours.
"""

import math
import random
import time


# ---------------------------------------------------------------------------
# I/O  (keep this stable — format is unforgiving)
# ---------------------------------------------------------------------------

def read_cities(path="input.txt"):
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    n = int(lines[0])
    cities = []
    for line in lines[1:1 + n]:
        x, y, z = map(int, line.split())
        cities.append((x, y, z))
    return cities


def write_output(tour, distance, path="output.txt"):
    with open(path, "w") as f:
        f.write("{:.3f}\n".format(distance))
        for city in tour:
            f.write("{} {} {}\n".format(city[0], city[1], city[2]))
        start = tour[0]
        f.write("{} {} {}\n".format(start[0], start[1], start[2]))


# ---------------------------------------------------------------------------
# Distance / fitness
# ---------------------------------------------------------------------------

def euclidean(a, b):
    return math.sqrt(
        (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
    )


def tour_distance(tour):
    """Closed-loop length: visit every city, then return to tour[0]."""
    total = 0.0
    n = len(tour)
    for i in range(n - 1):
        total += euclidean(tour[i], tour[i + 1])
    total += euclidean(tour[-1], tour[0])
    return total


def fitness(tour):
    """Higher is better. Inverse of distance (tiny epsilon avoids /0)."""
    return 1.0 / (tour_distance(tour) + 1e-12)


# ---------------------------------------------------------------------------
# Genetic Algorithm pieces (shapes match the homework tips)
# ---------------------------------------------------------------------------

def create_initial_population(size, cities):
    """
    CreateInitialPopulation(size, cities) -> list of tours

    Each individual is a random permutation of the cities.
    Also seeds a couple of "smarter" starts (nearest-neighbor) so the
    population isn't purely random garbage on larger N.
    """
    population = []

    # A few nearest-neighbor tours from different starting cities.
    nn_count = min(size // 4, len(cities))
    for i in range(nn_count):
        population.append(nearest_neighbor_tour(cities, start_index=i))

    # Fill the rest with random permutations.
    while len(population) < size:
        tour = list(cities)
        random.shuffle(tour)
        population.append(tour)

    return population


def nearest_neighbor_tour(cities, start_index=0):
    """Greedy heuristic: always jump to the closest unvisited city."""
    remaining = list(cities)
    tour = [remaining.pop(start_index % len(remaining))]
    while remaining:
        last = tour[-1]
        best_i = 0
        best_d = euclidean(last, remaining[0])
        for i in range(1, len(remaining)):
            d = euclidean(last, remaining[i])
            if d < best_d:
                best_d = d
                best_i = i
        tour.append(remaining.pop(best_i))
    return tour


def rank_population(population):
    """
    RankList: list of (index, fitness) sorted best-first (descending fitness).
    """
    ranked = []
    for i, tour in enumerate(population):
        ranked.append((i, fitness(tour)))
    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked


def create_mating_pool(population, rank_list):
    """
    CreateMatingPool(population, RankList) -> list of tours

    Roulette-wheel selection: fitness-proportional probabilities.
    Returns a mating pool the same size as the population.
    """
    total = 0.0
    for _, fit in rank_list:
        total += fit
    if total <= 0:
        return [list(t) for t in population]

    # Cumulative probability wheel.
    wheel = []
    running = 0.0
    for idx, fit in rank_list:
        running += fit / total
        wheel.append((running, idx))

    pool = []
    for _ in range(len(population)):
        r = random.random()
        chosen_idx = wheel[-1][1]
        for cutoff, idx in wheel:
            if r <= cutoff:
                chosen_idx = idx
                break
        pool.append(list(population[chosen_idx]))
    return pool


def crossover(parent1, parent2, start_index, end_index):
    """
    Crossover(Parent1, Parent2, Start_index, End_index) -> child tour

    Two-point / order crossover for TSP:
      1. Copy the slice parent1[start_index : end_index+1] into the child.
      2. Fill remaining slots from parent2 in order, skipping cities already
         present (so every city appears exactly once).

    Homework example idea:
      P1 = [1,2,3,4,5], P2 = [5,2,3,1,4], start=1, end=3
      slice from P1 = [2,3,4]
      child becomes [5,2,3,4,1]
    """
    n = len(parent1)
    if n == 0:
        return []
    if start_index > end_index:
        start_index, end_index = end_index, start_index
    start_index = max(0, min(start_index, n - 1))
    end_index = max(0, min(end_index, n - 1))

    child = [None] * n
    taken = set()

    # Step 1: copy the subarray from parent1.
    for i in range(start_index, end_index + 1):
        child[i] = parent1[i]
        taken.add(parent1[i])

    # Step 2: fill holes from parent2 in order.
    p2_pos = 0
    for i in range(n):
        if child[i] is not None:
            continue
        while p2_pos < n and parent2[p2_pos] in taken:
            p2_pos += 1
        child[i] = parent2[p2_pos]
        taken.add(parent2[p2_pos])
        p2_pos += 1

    return child


def mutate(tour, mutation_rate):
    """Swap-mutation: with probability mutation_rate, swap two random cities."""
    if len(tour) < 2:
        return tour
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(tour)), 2)
        tour[i], tour[j] = tour[j], tour[i]
    return tour


def breed_population(mating_pool, elite_size, mutation_rate):
    """
    Keep the elite_size best parents unchanged, then fill the rest by
    crossing over pairs from the mating pool and mutating the children.
    """
    # Elite: best individuals already sit at the front of mating_pool if we
    # pass a fitness-sorted pool — but roulette doesn't guarantee that, so
    # re-rank the pool for elitism.
    ranked = rank_population(mating_pool)
    next_gen = []
    for i in range(min(elite_size, len(mating_pool))):
        next_gen.append(list(mating_pool[ranked[i][0]]))

    while len(next_gen) < len(mating_pool):
        p1 = random.choice(mating_pool)
        p2 = random.choice(mating_pool)
        n = len(p1)
        if n >= 2:
            a, b = sorted(random.sample(range(n), 2))
        else:
            a, b = 0, 0
        child = crossover(p1, p2, a, b)
        child = mutate(child, mutation_rate)
        next_gen.append(child)

    return next_gen


def genetic_algorithm(cities, time_budget=None):
    """
    Main GA loop. Returns the best tour found.

    time_budget: seconds of wall time to spend (None => use a size heuristic).
    Vocareum kills long runs; leave headroom under the class limits
    (easy 60s / med 75s / hard 120s / complex 300s).
    """
    n = len(cities)
    if n == 0:
        return []
    if n == 1:
        return list(cities)

    # Scale population / generations loosely with problem size.
    if n <= 10:
        pop_size, generations, mutation_rate, elite_size = 80, 400, 0.05, 8
        default_budget = 5.0
    elif n <= 30:
        pop_size, generations, mutation_rate, elite_size = 120, 600, 0.08, 12
        default_budget = 40.0
    elif n <= 80:
        pop_size, generations, mutation_rate, elite_size = 160, 800, 0.10, 16
        default_budget = 90.0
    else:
        pop_size, generations, mutation_rate, elite_size = 200, 1200, 0.12, 20
        default_budget = 240.0

    if time_budget is None:
        time_budget = default_budget

    population = create_initial_population(pop_size, cities)
    best = min(population, key=tour_distance)
    best_dist = tour_distance(best)

    start = time.time()
    stagnant = 0

    for _gen in range(generations):
        if time.time() - start > time_budget:
            break

        ranked = rank_population(population)
        current = population[ranked[0][0]]
        current_dist = tour_distance(current)
        if current_dist + 1e-9 < best_dist:
            best = list(current)
            best_dist = current_dist
            stagnant = 0
        else:
            stagnant += 1

        # Early stop if we've plateaued for a while.
        if stagnant > 80:
            break

        mating_pool = create_mating_pool(population, ranked)
        population = breed_population(mating_pool, elite_size, mutation_rate)

    return best


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cities = read_cities("input.txt")
    tour = genetic_algorithm(cities)
    distance = tour_distance(tour)
    write_output(tour, distance, "output.txt")


if __name__ == "__main__":
    main()
