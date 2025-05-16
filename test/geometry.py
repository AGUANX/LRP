import numpy as np
import math

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def max_height_diff(p1, p2, height_map):
    h1 = height_map[p1] if p1[0] < height_map.shape[0] and p1[1] < height_map.shape[1] else 0
    h2 = height_map[p2] if p2[0] < height_map.shape[0] and p2[1] < height_map.shape[1] else 0
    return abs(h2 - h1)

def rotate_points(points, angle, center):
    angle_rad = math.radians(angle)
    R = np.array([
        [math.cos(angle_rad), -math.sin(angle_rad)],
        [math.sin(angle_rad), math.cos(angle_rad)]
    ])
    rotated_points = np.dot(points - center, R) + center
    return rotated_points