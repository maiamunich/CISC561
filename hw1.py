"""
HW1 I/O scaffold: read input.txt, emit a valid (naive) tour to output.txt.

City  = (x, y, z)  — three ints
Tour  = [city0, city1, ..., cityN-1]  — each city once; return-to-start is
         implied when computing distance / writing the extra closing line
"""

import math


def read_cities(path="input.txt"):
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    n = int(lines[0])
    cities = []
    for line in lines[1 : 1 + n]:
        x, y, z = map(int, line.split())
        cities.append((x, y, z))
    return cities


def euclidean(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def tour_distance(tour):
    """Closed-loop length: visit every city in order, then return to tour[0]."""
    total = 0.0
    for i in range(len(tour) - 1):
        total += euclidean(tour[i], tour[i + 1])
    total += euclidean(tour[-1], tour[0])
    return total


def write_output(tour, distance, path="output.txt"):
    with open(path, "w") as f:
        f.write("{:.3f}\n".format(distance))
        for city in tour:
            f.write("{} {} {}\n".format(city[0], city[1], city[2]))
        # Close the loop: last line must equal the first city.
        start = tour[0]
        f.write("{} {} {}\n".format(start[0], start[1], start[2]))


def main():
    cities = read_cities("input.txt")
    # Dumb but valid tour: cities in the order given.
    tour = list(cities)
    distance = tour_distance(tour)
    write_output(tour, distance, "output.txt")


if __name__ == "__main__":
    main()
