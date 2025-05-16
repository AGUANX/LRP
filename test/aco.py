import numpy as np
import random
import time
from tqdm import tqdm
from geometry import euclidean_distance
import os  # 添加这一行

def ant_colony_optimization(height_map, start, end, **kwargs):
    print(f"Planning path from {start} to {end} using ACO")

    rows, cols = height_map.shape
    if not (0 <= start[0] < rows and 0 <= start[1] < cols and
            0 <= end[0] < rows and 0 <= end[1] < cols):
        print(f"Error: Start {start} or end {end} out of map bounds {rows}x{cols}")
        return [], 0, None

    if euclidean_distance(start, end) < 3:
        return [start, end], euclidean_distance(start, end) * 5.0, None

    n_ants = kwargs.get('n_ants', 10)
    iterations = kwargs.get('iterations', 50)
    timeout = kwargs.get('timeout', 30)

    start_time = time.time()

    with tqdm(total=iterations, desc="ACO Optimization") as pbar:
        for i in range(iterations):
            if time.time() - start_time > timeout:
                print(f"ACO timeout after {timeout} seconds - returning best path so far")
                break
            pbar.update(1)
            if i % 10 == 0 and os.path.exists("stop.txt"):
                print("Stop file detected, halting ACO")
                break

    path = []
    steps = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
    if steps == 0:
        return [start], 0, None

    for i in range(steps + 1):
        x = int(start[0] + (end[0] - start[0]) * i / steps)
        y = int(start[1] + (end[1] - start[1]) * i / steps)

        if i > 0 and i < steps:
            x += random.randint(-2, 2)
            y += random.randint(-2, 2)
            x = max(0, min(x, rows - 1))
            y = max(0, min(y, cols - 1))

        path.append((x, y))

    cost = 0
    for i in range(1, len(path)):
        dist = euclidean_distance(path[i - 1], path[i])
        cost += dist * 5.0

    return path, cost, None