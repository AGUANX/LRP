import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

def visualize_mission(height_map, full_path, mission_areas, charging_stations, angles,
                      checkpoint_indices=[], recharge_indices=[], save_2d_path='mission_2d.png', save_3d_path='mission_3d.png'):
    try:
        # 2D 可视化
        plt.figure(figsize=(14, 12))

        plt.imshow(height_map, cmap='terrain', alpha=0.6)
        plt.colorbar(label='Elevation (m)')

        colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
        for i, mission_area in enumerate(mission_areas):
            x1, y1, x2, y2 = mission_area
            color = colors[i % len(colors)]
            rect = Rectangle((y1, x1), (y2 - y1), (x2 - x1),
                             linewidth=2, edgecolor=color, facecolor='none',
                             linestyle='--', label=f'Mission Area {i + 1}')
            plt.gca().add_patch(rect)

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            angle_rad = math.radians(angles[i])
            arrow_length = min(x2 - x1, y2 - y1) / 3
            dx = arrow_length * math.sin(angle_rad)
            dy = arrow_length * math.cos(angle_rad)
            plt.arrow(center_y, center_x, dy, dx, head_width=10, head_length=15,
                      fc=color, ec=color, label=f'Area {i + 1} Angle: {angles[i]}°')

        for i, station in enumerate(charging_stations):
            plt.scatter(station[1], station[0], color='yellow', marker='s', s=100,
                        edgecolor='black', label='Charging Station' if i == 0 else "")

        if full_path:
            stride = max(1, len(full_path) // 1000)
            sampled_path = full_path[::stride]

            path_x = [p[1] for p in sampled_path]
            path_y = [p[0] for p in sampled_path]
            plt.plot(path_x, path_y, 'gray', linewidth=0.8, alpha=0.7, label='Flight Path')

            plt.scatter(full_path[0][1], full_path[0][0], color='green', marker='o', s=100, label='Start')
            plt.scatter(full_path[-1][1], full_path[-1][0], color='red', marker='x', s=100, label='End')

            for idx in checkpoint_indices:
                if 0 <= idx < len(full_path):
                    plt.scatter(full_path[idx][1], full_path[idx][0], color='blue', marker='*', s=80,
                                label='Mission Checkpoint' if idx == checkpoint_indices[0] else "")

            for idx in recharge_indices:
                if 0 <= idx < len(full_path):
                    plt.scatter(full_path[idx][1], full_path[idx][0], color='orange', marker='^', s=80,
                                label='Recharge Point' if idx == recharge_indices[0] else "")

        plt.title('Complete UAV Mission Visualization', fontsize=16)
        plt.xlabel('X')
        plt.ylabel('Y')

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_2d_path, dpi=150)
        plt.close()
        print(f"2D visualization saved to {save_2d_path}")

        # 3D 可视化
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')

        sample_rate = 10
        y, x = np.mgrid[0:height_map.shape[0]:sample_rate, 0:height_map.shape[1]:sample_rate]
        z = height_map[::sample_rate, ::sample_rate]
        z = np.nan_to_num(z, nan=np.nanmean(z))

        surf = ax.plot_surface(x, y, z, cmap='terrain', alpha=0.6, linewidth=0, antialiased=True)

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Elevation (m)')

        colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
        for i, area in enumerate(mission_areas):
            x1, y1, x2, y2 = area
            color = colors[i % len(colors)]

            corners_x = [y1, y2, y2, y1, y1]
            corners_y = [x1, x1, x2, x2, x1]

            corners_z = []
            for y, x in zip(corners_y, corners_x):
                if 0 <= y < height_map.shape[0] and 0 <= x < height_map.shape[1]:
                    height = height_map[y, x]
                    if np.isnan(height):
                        height = np.nanmean(height_map)
                    corners_z.append(height)
                else:
                    corners_z.append(np.nanmean(height_map))

            ax.plot(corners_x, corners_y, corners_z, color=color, linestyle='--', linewidth=2,
                    label=f'Mission Area {i + 1}')

        for i, station in enumerate(charging_stations):
            x, y = station
            if 0 <= x < height_map.shape[0] and 0 <= y < height_map.shape[1]:
                z = height_map[x, y]
                if np.isnan(z):
                    z = np.nanmean(height_map)
                ax.scatter([y], [x], [z], color='yellow', marker='s', s=100, edgecolor='black',
                           label='Charging Station' if i == 0 else "")

        if full_path:
            stride = max(1, len(full_path) // 100)
            path_reduced = full_path[::stride]

            path_x = [p[1] for p in path_reduced]
            path_y = [p[0] for p in path_reduced]
            path_z = []

            for y, x in zip(path_y, path_x):
                if 0 <= y < height_map.shape[0] and 0 <= x < height_map.shape[1]:
                    z_val = height_map[y, x]
                    if np.isnan(z_val):
                        z_val = np.nanmean(height_map)
                    path_z.append(z_val)
                else:
                    path_z.append(np.nanmean(height_map))

            ax.plot(path_x, path_y, path_z, 'gray', linewidth=1.5, label='Flight Path')

            if full_path:
                start = full_path[0]
                end = full_path[-1]

                if 0 <= start[0] < height_map.shape[0] and 0 <= start[1] < height_map.shape[1]:
                    start_z = height_map[start[0], start[1]]
                    if np.isnan(start_z):
                        start_z = np.nanmean(height_map)
                    ax.scatter(start[1], start[0], start_z,
                               color='green', marker='o', s=100, label='Start')

                if 0 <= end[0] < height_map.shape[0] and 0 <= end[1] < height_map.shape[1]:
                    end_z = height_map[end[0], end[1]]
                    if np.isnan(end_z):
                        end_z = np.nanmean(height_map)
                    ax.scatter(end[1], end[0], end_z,
                               color='red', marker='x', s=100, label='End')

        ax.set_title('3D UAV Mission Visualization', fontsize=16)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Elevation (m)')

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        ax.view_init(elev=30, azim=135)

        plt.tight_layout()
        plt.savefig(save_3d_path, dpi=150)
        plt.close()
        print(f"3D visualization saved to {save_3d_path}")

    except Exception as e:
        print(f"Error in visualization: {e}")