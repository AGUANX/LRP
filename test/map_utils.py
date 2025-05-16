import numpy as np

def read_map_from_csv(file_path):
    try:
        return np.loadtxt(file_path, delimiter=',')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return np.random.normal(1000, 300, (600, 600))

def is_within_map(point, height_map):
    x, y = point
    return 0 <= x < height_map.shape[0] and 0 <= y < height_map.shape[1]

def find_valid_start_point(height_map):
    rows, cols = height_map.shape
    for i in range(rows):
        for j in range(cols):
            if not np.isnan(height_map[i, j]):
                return (i, j)
    return None