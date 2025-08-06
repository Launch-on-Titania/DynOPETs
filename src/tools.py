import cv2
import os
import numpy as np
import open3d as o3d
from pathlib import Path
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from rich.progress import track
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from scipy.spatial.transform import Rotation as R

current_file_path = Path(__file__).resolve()
project_dir = current_file_path.parent.parent
DATASET_DIR = "~/Datasets/DynOPETs/COPE119"


# ============================================================================
# File I/O Functions
# ============================================================================

def load_tum_poses(file_path):
    """
    Load poses from a TUM format file.
    Returns:
        timestamps: list of float
        poses: list of (R, t), where R is (3,3) rotation matrix, t is (3,) translation
    """
    timestamps = []
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip() == "":
                continue
            vals = line.strip().split()
            if len(vals) != 8:
                continue
            ts = float(vals[0])
            tx, ty, tz = map(float, vals[1:4])
            qx, qy, qz, qw = map(float, vals[4:8])
            rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
            t = np.array([tx, ty, tz])
            pose = np.eye(4)
            pose[:3, :3] = rot
            pose[:3, 3] = t
            
            timestamps.append(ts)
            poses.append(pose)
            
    return timestamps, poses


def save_tum_poses(file_path, timestamps, poses):
    """
    Save poses to a TUM format file.
    poses: list of (R, t), where R is (3,3) rotation matrix, t is (3,) translation
    """
    with open(file_path, 'w') as f:
        for ts, T in zip(timestamps, poses):
            t = T[:3, 3]
            rot = T[:3, :3]
            quat = R.from_matrix(rot).as_quat()  # x, y, z, w
            f.write(f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n")


def load_model(model_path):
    """Load 3D model from file."""
    points = None
    if model_path.endswith(".obj"):
        mesh = o3d.io.read_triangle_mesh(model_path)
        points = np.asarray(mesh.vertices)
    elif model_path.endswith(".ply"):
        points = np.asarray(o3d.io.read_point_cloud(model_path).points)
    else:
        raise ValueError("Unsupported model format")
    return points


def get_model_scale(model_path):
    """Get model scale matrix."""
    model_points = load_model(model_path)
    scales = np.max(np.abs(model_points), axis=0)
    return np.diag(scales)


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_cube_with_transform(image, camera_matrix, dist_coeffs, transform, scale, color=(0, 255, 0)):
    """Plot a cube with given transformation on the image."""
    # Vertices of a unit cube
    vertices = np.array([
        [0, 0, 0],  # Vertex 0
        [1, 0, 0],  # Vertex 1
        [1, 1, 0],  # Vertex 2
        [0, 1, 0],  # Vertex 3
        [0, 0, 1],  # Vertex 4
        [1, 0, 1],  # Vertex 5
        [1, 1, 1],  # Vertex 6
        [0, 1, 1],  # Vertex 7
    ])
    vertices = vertices * 2 - 1
    vertices = vertices @ scale.T
    
    # Edges of the unit cube, each represented by a pair of vertices
    edges = np.array([
        [vertices[0], vertices[1]],
        [vertices[1], vertices[2]],
        [vertices[2], vertices[3]],
        [vertices[3], vertices[0]],
        [vertices[4], vertices[5]],
        [vertices[5], vertices[6]],
        [vertices[6], vertices[7]],
        [vertices[7], vertices[4]],
        [vertices[0], vertices[4]],
        [vertices[1], vertices[5]],
        [vertices[2], vertices[6]],
        [vertices[3], vertices[7]],
    ])
    
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    tvec = transform[:3, 3].reshape(1, 3)
    
    for i in range(12):
        object_points = edges[i].reshape(2, 3)
        points2d, _ = cv2.projectPoints(
            object_points, rvec, tvec, camera_matrix, dist_coeffs
        )
        points2d = points2d.reshape(2, 2)
        image = cv2.line(
            image,
            (int(points2d[0][0]), int(points2d[0][1])),
            (int(points2d[1][0]), int(points2d[1][1])),
            color,
            thickness=3,
            lineType=cv2.LINE_AA,
        )
    return image


def generate_trajectory(points, color):
    """Generate trajectory line collection for 3D plotting."""
    segments = [(i, j) for i, j in zip(points[:-1], points[1:])]
    trajectory = Line3DCollection(segments, colors=color, linewidths=1)
    return trajectory


def plot_3d_trajectory(c2o_fd_files, parent_path, color="blue", traj_name="0"):
    """Plot 3D trajectory from pose files."""
    t_fdpose = [np.zeros(3)]
    num_fdpose_files = len(c2o_fd_files)

    T_c2o_0 = np.loadtxt(os.path.join(parent_path, c2o_fd_files[0]))
    for i in track(
        range(1, num_fdpose_files), description="show Foundation Pose Trajectory"
    ):
        T_c2o_i = np.loadtxt(os.path.join(parent_path, c2o_fd_files[i]))
        T_ci2c0_fdpose = (
            np.linalg.inv(T_c2o_0) @ T_c2o_i
        )  # T_ci2cj is the transformation from frame i to frame j(i+1)
        t_fdpose.append(T_ci2c0_fdpose[:3, 3])
    t_fdpose = np.array(t_fdpose).reshape(-1, 3).T

    fig = plt.figure()
    # set fdpose in the middle of the coordinate
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(t_fdpose[0, :].min() - 1, t_fdpose[0, :].max() + 1)
    ax.set_ylim(t_fdpose[1, :].min() - 1, t_fdpose[1, :].max() + 1)
    ax.set_zlim(t_fdpose[2, :].min() - 1, t_fdpose[2, :].max() + 1)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    lc_manual = generate_trajectory(t_fdpose.T, color)
    ax.add_collection(lc_manual)
    ax.set_box_aspect((1, 1, 1))

    plt.title(f"{traj_name} Trajectory")
    plt.savefig(f"{traj_name}.svg")
    plt.show()


def print_comparison(ori_pose, opt_pose):
    """Print comparison of original and optimized poses using rich console."""
    console = Console()
    left_content = str(ori_pose)
    right_content = str(opt_pose)

    layout = Layout()
    layout.split_row(Layout(name="left", ratio=1), Layout(name="right", ratio=1))
    layout["left"].update(
        Panel(
            Text(left_content, overflow="fold", no_wrap=True, style='cyan'),
            title="[bold blue]Original", width=55, height=6
        )
    )
    layout["right"].update(
        Panel(
            Text(right_content, overflow="fold", no_wrap=True, style='green'),
            title="[bold red]Refined", width=55, height=6
        )
    )

    console.print(layout)
    print('[bold red]---------------------------------------------------------------------')


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_comparison_curve(data1, data2, fps=15):
    """Plot comparison curve between two datasets."""
    if len(data1) != len(data2):
        raise ValueError("Data lengths do not match.")

    time_data = list(range(len(data1)))
    time_in_seconds = np.array(time_data) / fps

    plt.figure(figsize=(10, 6))
    plt.plot(time_in_seconds, data1, label='raw', color='b')
    plt.plot(time_in_seconds, data2, label='ekf', color='r')

    plt.title('Comparison of Two Data Sets Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Value')

    plt.legend()
    plt.grid(True)
    plt.show()


def plot_original_rotation(log_dir, data_dir, data1, data2, data3, fps=15):
    """Plot original rotation data (yaw, pitch, roll)."""
    time_data = np.array(range(len(data1[0])))
    time_in_seconds = np.array(time_data) / fps

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    axes[0].plot(time_in_seconds, data1[0], label='Yaw raw - A', color='r', linestyle='-')
    axes[0].set_title('Yaw')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(time_in_seconds, data2[0], label='Pitch raw - A', color='r', linestyle='-')
    axes[1].set_title('Pitch')
    axes[1].set_ylabel('Value')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(time_in_seconds, data3[0], label='Roll raw - A', color='r', linestyle='-')
    axes[2].set_title('Roll')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('Value')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    order = data_dir.split("/")[-1]
    plt.savefig(f'{log_dir}/{order}_rot.png')


def plot_original_translation(log_dir, data_dir, data1, data2, data3, fps=15):
    """Plot original translation data (X, Y, Z)."""
    time_data = np.array(range(len(data1[0])))
    time_in_seconds = np.array(time_data) / fps

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    axes[0].plot(time_in_seconds, data1[0], label='X raw - A', color='r', linestyle='-')
    axes[0].set_title('X')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(time_in_seconds, data2[0], label='Y raw - A', color='r', linestyle='-')
    axes[1].set_title('Y')
    axes[1].set_ylabel('Value')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(time_in_seconds, data3[0], label='Z raw - A', color='r', linestyle='-')
    axes[2].set_title('Z')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('Value')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    order = data_dir.split("/")[-1]
    plt.savefig(f'{log_dir}/{order}_rot.png')


def plot_three_pairs_comparison_subplots(log_dir, source_dir, data1, data2, data3, fps=15, show=False):
    """Plot three pairs comparison in subplots (rotation data)."""
    time_data = np.array(range(len(data1[0])))
    time_in_seconds = np.array(time_data) / fps

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    axes[0].plot(time_in_seconds, data1[0], label='Yaw raw - A', color='b', linestyle='-')
    axes[0].plot(time_in_seconds, data1[1], label='Yaw ekf - B', color='r', linestyle='-')
    axes[0].plot(time_in_seconds, data1[2], label='Yaw global ekf - C', color='g', linestyle='-')
    axes[0].set_title('Yaw')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(time_in_seconds, data2[0], label='Pitch raw - A', color='b', linestyle='-')
    axes[1].plot(time_in_seconds, data2[1], label='Pitch ekf - B', color='r', linestyle='-')
    axes[1].plot(time_in_seconds, data2[2], label='Pitch global ekf - C', color='g', linestyle='-')
    axes[1].set_title('Pitch')
    axes[1].set_ylabel('Value')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(time_in_seconds, data3[0], label='Roll raw - A', color='b', linestyle='-')
    axes[2].plot(time_in_seconds, data3[1], label='Roll ekf - B', color='r', linestyle='-')
    axes[2].plot(time_in_seconds, data3[2], label='Roll global ekf - C', color='g', linestyle='-')
    axes[2].set_title('Roll')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('Value')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    save_path = os.path.join(log_dir, f"{source_dir}_global_rotation_comparison.png")
    plt.savefig(save_path)
    
    if show:
        plt.show()
        plt.close()


def plot_translation_comparison(log_dir, source_dir, x_pairs, y_pairs, z_pairs, show=False):
    """Plot translation comparison (X, Y, Z components)."""
    time = np.arange(len(x_pairs[0]))
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(time, x_pairs[0], label="Raw X", color="blue")
    plt.plot(time, x_pairs[1], label="Ekf Smoothed X", color="red")
    plt.plot(time, x_pairs[2], label="Global Smoothed X", color="green")
    plt.title("Translation X Comparison")
    plt.xlabel("Frame")
    plt.ylabel("X (meters)")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(time, y_pairs[0], label="Raw Y", color="blue")
    plt.plot(time, y_pairs[1], label="Smoothed Y", color="red")
    plt.plot(time, y_pairs[2], label="Global Smoothed Y", color="green")
    plt.title("Translation Y Comparison")
    plt.xlabel("Frame")
    plt.ylabel("Y (meters)")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(time, z_pairs[0], label="Raw Z", color="blue")
    plt.plot(time, z_pairs[1], label="Smoothed Z", color="red")
    plt.plot(time, z_pairs[2], label="Global Smoothed Z", color="green")
    plt.title("Translation Z Comparison")
    plt.xlabel("Frame")
    plt.ylabel("Z (meters)")
    plt.legend()

    plt.tight_layout()
    save_path = os.path.join(log_dir, f"{source_dir}_global_translation_comparison.png")
    plt.savefig(save_path)
    
    if show:
        plt.show()
        plt.close()