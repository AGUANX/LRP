import numpy as np
import random

def generate_mission_areas(height_map, num_mission_areas=2, area_size_range=(60, 120)):
    rows, cols = height_map.shape
    mission_areas = []

    max_area_size = min(area_size_range[1], min(rows, cols) // 3)
    min_area_size = min(area_size_range[0], max_area_size // 2)

    for i in range(num_mission_areas):
        area_size = random.randint(min_area_size, max_area_size)

        max_attempts = 20
        for attempt in range(max_attempts):
            x1 = random.randint(10, rows - area_size - 10)
            y1 = random.randint(10, cols - area_size - 10)
            x2 = x1 + area_size
            y2 = y1 + area_size

            x2 = min(x2, rows - 1)
            y2 = min(y2, cols - 1)

            overlap = False
            for existing_area in mission_areas:
                ex1, ey1, ex2, ey2 = existing_area
                if (x1 <= ex2 and x2 >= ex1 and y1 <= ey2 and y2 >= ey1):
                    overlap = True
                    break

            if not overlap:
                area_valid = False
                for check_x in range(x1, x2, max(1, (x2 - x1) // 5)):
                    for check_y in range(y1, y2, max(1, (y2 - y1) // 5)):
                        if 0 <= check_x < rows and 0 <= check_y < cols:
                            if not np.isnan(height_map[check_x, check_y]):
                                area_valid = True
                                break
                    if area_valid:
                        break

                if area_valid:
                    mission_areas.append((x1, y1, x2, y2))
                    break

    validated_mission_areas = []
    for area in mission_areas:
        x1, y1, x2, y2 = area
        if (0 <= x1 < rows and 0 <= y1 < cols and
                0 <= x2 < rows and 0 <= y2 < cols and
                not np.isnan(height_map[x1, y1])):
            validated_mission_areas.append(area)

    return validated_mission_areas

def calculate_optimal_angles(num_areas):
    predefined_angles = [0, 45, 90, 135]
    angles = []
    for i in range(num_areas):
        angle = predefined_angles[i % len(predefined_angles)]
        angles.append(angle)
    return angles