"""
HW1 I/O scaffold: read input.txt, emit a valid (naive) tour to output.txt.

City  = (x, y, z)  — three ints
Tour  = [city0, city1, ..., cityN-1]  — each city once; return-to-start is
         implied when computing distance / writing the extra closing line
"""

import math
import random
from typing import Any


def read_cities(path="input.txt"):
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    n = int(lines[0])
    cities = []
    for line in lines[1 : 1 + n]:
        x, y, z = map(int, line.split())
        cities.append((x, y, z))
    return cities

def write_output(tour, distance, path="output.txt"):
    with open(path, "w") as f:
        f.write("{:.3f}\n".format(distance))
        for city in tour:
            f.write("{} {} {}\n".format(city[0], city[1], city[2]))
        # Close the loop: last line must equal the first city.
        start = tour[0]
        f.write("{} {} {}\n".format(start[0], start[1], start[2]))

def euclidean(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def tour_distance(tour):
    """Closed-loop length: visit every city in order, then return to tour[0]."""
    total = 0.0
    for i in range(len(tour) - 1):
        total += euclidean(tour[i], tour[i + 1])
    total += euclidean(tour[-1], tour[0])
    return total

def chooose_size_population(cities):
    if len(cities) <= 50:
        return len(cities) * 2
    elif len(cities) <= 200:
        return len(cities)
    else:
        return 100

def create_initial_population(cities):
    size = chooose_size_population(cities)
    population = []
    for i in range(size):
        tour = list(cities)
        random.shuffle(tour)
        population.append(tour)
    return population

def evaluate_population(population):
    fitness = []
    for i in range(len(population)):
        distance = tour_distance(population[i])
        fitness.append(distance)
    return fitness


def order(fitness, population):
    paired = sorted(zip(fitness, population), key=lambda pair: pair[0])
    ordered_fitness = [f for f, _ in paired]
    ordered_population = [tour for _, tour in paired]
    return ordered_fitness, ordered_population

def repair(child, all_cities):
    child = list(child)
    seen = set()
    duplicate_indices = []
    for i, city in enumerate(child):
        if city in seen:
            duplicate_indices.append(i)
        else:
            seen.add(city)

    missing = [city for city in all_cities if city not in seen]
    for i, city in zip(duplicate_indices, missing):
        child[i] = city
    return child


def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child = parent1[0:crossover_point] + parent2[crossover_point:]
    return repair(child, parent1)

def crossover2(parent1, parent2, start, end):
    child = parent2 
    child[start:end] = parent1[start:end]
    return repair(child, parent1)

def mutate(tour):
    tour = list(tour)
    i, j = random.sample(range(len(tour)), 2)
    tour[i], tour[j] = tour[j], tour[i]
    return tour

def genetic_algorithm(cities, max_generations=100):
    population = create_initial_population(cities)
    fitness = evaluate_population(population)
    fitness, population = order(fitness, population)
    generation = 0
    while True:
        new_population = []
        for i in range(len(population)):
            parent1 = population[i]
            parent2 = population[(i + 1) % len(population)]
            start = random.randint(1, len(parent1) - 1)
            end = random.randint(1, len(parent1) - 1)
            if start > end:
                start, end = end, start
            child = crossover2(parent1, parent2, start, end)
            new_population.append(child)
        fitness = evaluate_population(new_population)
        fitness, population = order(fitness, new_population)
        generation += 1
        if generation >= max_generations:
            break

    return population[0]

def main():
    cities = read_cities("input.txt")
    tour = genetic_algorithm(cities, max_generations=100)
    distance = tour_distance(tour)
    write_output(tour, distance, "output.txt")


if __name__ == "__main__":
    main()
