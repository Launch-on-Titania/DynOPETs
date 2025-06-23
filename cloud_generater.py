import numpy as np
import cv2
from pathlib import Path
import open3d as o3d
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def process_image(i, mask_paths, rgb_paths, depth_paths, output_path, fx, fy, cx, cy):
    """Process a single image to generate point cloud."""
    mask_path = mask_paths / f"{i:04d}_mask.png"
    rgb_path = rgb_paths / f"{i:04d}_color.png"
    depth_path = depth_paths / f"{i:04d}_depth.png"

    # Load images
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(float)

    # Validate image dimensions
    assert mask.shape == rgb.shape[:2] == depth.shape
    
    # Preprocess images
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(float) / 255.0
    mask = mask < 100
    
    # Generate 3D points
    indices = np.dstack(np.indices(mask.shape))
    u = indices[:,:,1]
    v = indices[:,:,0]

    z = depth[mask]
    x = (u[mask] - cx) * z / fx
    y = (v[mask] - cy) * z / fy

    points = np.vstack((x, y, z)).T / 1000.0  # Convert to meters
    colors = rgb[mask]

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Remove outliers from the point cloud
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=2.0)

    # Save point cloud
    o3d.io.write_point_cloud(str(output_path / f"{i:04d}.ply"), pcd)


def generate_point_clouds(data_dir, fx, fy, cx, cy):
    """Generate point clouds for all images in the dataset."""
    mask_paths = data_dir / "mask"
    rgb_paths = data_dir / "color"
    depth_paths = data_dir / "depth"
    output_path = data_dir / "clouds"

    # Create output directory
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    # Count number of files to process
    num_files = len([f for f in rgb_paths.iterdir() if f.is_file()])
    
    # Process images in parallel
    def process_wrapper(i):
        return process_image(i, mask_paths, rgb_paths, depth_paths, output_path, fx, fy, cx, cy)
    
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_wrapper, range(num_files)), total=num_files))


def main():
    parser = argparse.ArgumentParser(description="Generate point clouds from RGB-D images")
    parser.add_argument("--subset", type=str, default="COPE119", help="Dataset subset")
    parser.add_argument("--seqs", type=str, default="bottle_00", help="Sequence name")
    args = parser.parse_args()
    
    # Setup paths
    root_dir = "~/Datasets/DynOPETs"
    data_dir = Path(root_dir).expanduser() / args.subset / args.seqs.split('_')[0] / args.seqs

    # Load camera intrinsics
    fx, fy, cx, cy, _, _, _, _ = np.loadtxt(data_dir / "intrinsics.txt")

    # Generate point clouds
    generate_point_clouds(data_dir, fx, fy, cx, cy)


if __name__ == "__main__":
    main()