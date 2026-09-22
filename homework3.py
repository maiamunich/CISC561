import math
import random
import time

#the time limit is used for the generic algorithm 
#(instead of running 100 generations like my old code it 
#runs for 45s allowing for it to find a better tour)
TIME_LIMIT = 45.0

# this reads the cities from the input files 
def read_cities(path="input.txt"):
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    n = int(lines[0])
    cities = []
    for line in lines[1 : 1 + n]:
        x, y, z = map(int, line.split())
        cities.append((x, y, z))
    return cities

# this reads out the tour and distance to the output file
def write_output(tour, distance, path="output.txt"):
    with open(path, "w") as f:
        f.write("{:.3f}\n".format(distance))
        for city in tour:
            f.write("{} {} {}\n".format(city[0], city[1], city[2]))
        start = tour[0]
        f.write("{} {} {}\n".format(start[0], start[1], start[2]))

# this calculates the euclidean distance between two cities
def euclidean(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

# this builds the distance matrix (distance matrix is a square matrix of size n x n, where n is the number of cities) for the cities
def build_distance_matrix(cities):
    n = len(cities)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean(cities[i], cities[j])
            matrix[i][j] = d
            matrix[j][i] = d
    return matrix

# this calculates the total distance of a tour by summing the distances between consecutive cities
def tour_distance(tour, dist):
    total = 0.0
    for i in range(len(tour) - 1):
        total += dist[tour[i]][tour[i + 1]]
    total += dist[tour[-1]][tour[0]]
    return total

# this chooses the size of the population based on the number of cities
def chooose_size_population(cities):
    n = len(cities)
    if n <= 100:
        return 24
    elif n <= 400:
        return 16
    else:
        return 10

# this implements the nearest neighbor algorithm to find the initial tour
def nearest_neighbor(start, n, dist):
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    current = start
    while unvisited:
        best = None
        best_d = float("inf")
        for j in unvisited:
            d = dist[current][j]
            if d < best_d:
                best_d = d
                best = j
        unvisited.remove(best)
        tour.append(best)
        current = best
    return tour

# this builds the neighbors list for each city (neighbors list is a list of cities that are the nearest neighbors of the city)
def build_neighbors(n, dist, k=12):
    neighbors = []
    for i in range(n):
        order = sorted(range(n), key=lambda j: dist[i][j] if j != i else float("inf"))
        neighbors.append(order[:k])
    return neighbors

# this implements the 2-opt algorithm to improve the tour
def two_opt(tour, dist, neighbors, deadline):
    n = len(tour)
    pos = [0] * n
    for idx, city in enumerate(tour):
        pos[city] = idx

    improved = True
    while improved:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                return tour
            a = tour[i]
            a_next = tour[(i + 1) % n]
            d_a = dist[a][a_next]
            for c in neighbors[a]:
                d_ac = dist[a][c]
                if d_ac >= d_a:
                    break
                j = pos[c]
                c_next = tour[(j + 1) % n]
                if c_next == a or c == a_next:
                    continue
                delta = d_ac + dist[a_next][c_next] - d_a - dist[c][c_next]
                if delta < -1e-10:
                    low, high = i + 1, j
                    if low > high:
                        low, high = j + 1, i
                    while low < high:
                        tour[low], tour[high] = tour[high], tour[low]
                        pos[tour[low]] = low
                        pos[tour[high]] = high
                        low += 1
                        high -= 1
                    improved = True
                    a_next = tour[(i + 1) % n]
                    d_a = dist[a][a_next]
    return tour

# this creates the initial population of tours by using the nearest neighbor algorithm and the 2-opt algorithm
def create_initial_population(cities, dist, neighbors, deadline):
    size = chooose_size_population(cities)
    n = len(cities)
    population = []

    starts = list(range(n))
    random.shuffle(starts)
    for start in starts[:size]:
        if time.time() > deadline:
            break
        tour = nearest_neighbor(start, n, dist)
        tour = two_opt(tour, dist, neighbors, deadline)
        population.append(tour)

    while len(population) < size:
        tour = list(range(n))
        random.shuffle(tour)
        tour = two_opt(tour, dist, neighbors, deadline)
        population.append(tour)
        if time.time() > deadline:
            break

    if not population:
        population = [list(range(n))]
    return population

# this evaluates the fitness of the population by calculating the total distance of each tour
def evaluate_population(population, dist):
    fitness = []
    for i in range(len(population)):
        distance = tour_distance(population[i], dist)
        fitness.append(distance)
    return fitness

#this orders the population by fitness
def order(fitness, population):
    paired = sorted(zip(fitness, population), key=lambda pair: pair[0])
    ordered_fitness = [f for f, _ in paired]
    ordered_population = [tour for _, tour in paired]
    return ordered_fitness, ordered_population

#this repairs the child tour by removing duplicate cities
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

#this implements the crossover operation to create a child tour by combining two parent tours
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child = parent1[0:crossover_point] + parent2[crossover_point:]
    return repair(child, list(range(len(parent1))))

#this implements the crossover operation to create a child tour by combining two parent tours using the order crossover method
def crossover2(parent1, parent2, start, end, all_cities):
    n = len(parent1)
    child = [-1] * n
    child[start : end + 1] = parent1[start : end + 1]
    taken = set(parent1[start : end + 1])
    filler = [city for city in parent2 if city not in taken]
    k = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = filler[k]
            k += 1
    return child

#this implements the mutation operation to create a new tour by swapping two cities
def mutate(tour, dist=None, neighbors=None, deadline=None):
    tour = list(tour)
    n = len(tour)

    if neighbors is not None and dist is not None and deadline is not None:
        if random.random() < 0.3 and n >= 8:
            a, b, c = sorted(random.sample(range(1, n), 3))
            tour = tour[:a] + tour[b:c] + tour[a:b] + tour[c:]
        elif random.random() < 0.2:
            i, j = random.sample(range(n), 2)
            tour[i], tour[j] = tour[j], tour[i]
        tour = two_opt(tour, dist, neighbors, deadline)
        return tour

    i, j = random.sample(range(n), 2)
    tour[i], tour[j] = tour[j], tour[i]
    return tour

#this implements the genetic algorithm to find the best tour
def genetic_algorithm(cities, dist, neighbors, deadline):
    n = len(cities)
    if n <= 3:
        return list(range(n))

    all_indices = list(range(n))
    population = create_initial_population(cities, dist, neighbors, deadline)
    fitness = evaluate_population(population, dist)
    fitness, population = order(fitness, population)

    best_tour = list(population[0])
    best_fit = fitness[0]

    while time.time() < deadline:
        size = len(population)
        elite_count = max(1, int(size * 0.10))
        if elite_count < 2 and size >= 2:
            elite_count = 2

        new_population = [list(population[i]) for i in range(elite_count)]

        while len(new_population) < size and time.time() < deadline:
            cutoff = max(elite_count, size // 2)
            i = random.randrange(cutoff)
            j = random.randrange(cutoff)
            parent1 = population[i]
            parent2 = population[j]

            start = random.randint(0, n - 1)
            end = random.randint(0, n - 1)
            if start > end:
                start, end = end, start
            child = crossover2(parent1, parent2, start, end, all_indices)

            if random.random() < 0.4:
                child = mutate(child, dist, neighbors, deadline)
            else:
                child = two_opt(child, dist, neighbors, deadline)

            new_population.append(child)

        fitness = evaluate_population(new_population, dist)
        fitness, population = order(fitness, new_population)

        if fitness[0] < best_fit - 1e-9:
            best_fit = fitness[0]
            best_tour = list(population[0])
        else:
            kicked = mutate(best_tour, dist, neighbors, deadline)
            population[-1] = kicked
            fitness[-1] = tour_distance(kicked, dist)
            fitness, population = order(fitness, population)
            if fitness[0] < best_fit - 1e-9:
                best_fit = fitness[0]
                best_tour = list(population[0])

    return best_tour

#this is the main function that reads the cities from the input file, builds the distance matrix and neighbors list, and runs the genetic algorithm to find the best tour
def main():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    cities = read_cities("input.txt")
    n = len(cities)
    dist = build_distance_matrix(cities)
    neighbors = build_neighbors(n, dist, k=min(12, max(1, n - 1)))

    best_indices = genetic_algorithm(cities, dist, neighbors, deadline)
    tour = [cities[i] for i in best_indices]
    distance = tour_distance(best_indices, dist)
    write_output(tour, distance, "output.txt")


if __name__ == "__main__":
    main()
