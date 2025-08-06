import torch
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation


class RelativePoseEstimator:
    """
    Relative pose estimation between two frames using feature tracks and depth information.
    
    This class implements RANSAC-based pose estimation with Arun's method for 3D-3D alignment,
    including covariance estimation for uncertainty quantification.
    """
    
    def __init__(self, 
                 ransac_iterations: int = 100,
                 min_points: int = 3,
                 inlier_threshold: float = 0.1,
                 min_inlier_ratio: float = 0.5,
                 numerical_eps: float = 1e-9,
                 ):
        """
        Initialize the relative pose estimator.
        
        Args:
            ransac_iterations: Number of RANSAC iterations
            min_points: Minimum points needed for pose estimation
            inlier_threshold: Threshold for considering a point as an inlier (in meters)
            min_inlier_ratio: Minimum ratio of inliers to total points
            numerical_eps: Small perturbation for numerical differentiation
        """
        self.ransac_iterations = ransac_iterations
        self.min_points = min_points
        self.inlier_threshold = inlier_threshold
        self.min_inlier_ratio = min_inlier_ratio
        self.numerical_eps = numerical_eps
        self.H = H
        self.W = W
        
        # VGG-SfM specific scaling parameters (can be made configurable)
        self.x_scale = self.W / 1024
        self.y_scale = self.H / 1024
        self.y_shift = -(self.H - self.W) / 2
    
    def estimate_pose(self, 
                     tracks: np.ndarray, 
                     intrinsics: np.ndarray, 
                     depth_images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate relative pose between two frames.
        
        Args:
            tracks: Feature tracks of shape (2, N, 2) - 2D coordinates for N points in 2 frames
            intrinsics: Camera intrinsic matrix of shape (3, 3)
            depth_images: Depth images of shape (2, H, W)
            
        Returns:
            Tuple of (transformation_matrix, covariance_matrix)
            - transformation_matrix: 4x4 transformation matrix from frame 0 to frame 1
            - covariance_matrix: 6x6 covariance matrix for pose uncertainty
        """
        # Convert 2D tracks to 3D points
        points_3d = self._tracks_to_3d_points(tracks, intrinsics, depth_images)
        
        # Filter valid points
        source_points, target_points = self._filter_valid_points(points_3d)
        
        # RANSAC pose estimation
        best_transform, best_inlier_mask = self._ransac_pose_estimation(
            source_points, target_points)
        
        # Compute covariance matrix
        covariance = self._compute_covariance(
            source_points, target_points, best_transform, best_inlier_mask)
        
        return best_transform, covariance
    
    def _tracks_to_3d_points(self, 
                            tracks: np.ndarray, 
                            intrinsics: np.ndarray, 
                            depth_images: np.ndarray) -> torch.Tensor:
        """Convert 2D tracks to 3D points using depth and camera intrinsics."""
        tracks = torch.from_numpy(tracks)
        depth_images = torch.from_numpy(depth_images)
        intrinsics = torch.from_numpy(intrinsics)
        
        points_3d = []
        for frame_idx in range(2):
            points_2d = tracks[frame_idx]  # (N, 2)
            
            # Rescale coordinates from VGG-SfM output
            points_2d_rescaled = self._rescale_coordinates(points_2d)
            
            # Sample depths using bilinear interpolation
            depths = self._sample_depths(points_2d_rescaled, depth_images[frame_idx])
            
            # Unproject to 3D
            points_3d_frame = self._unproject_points(points_2d_rescaled, depths, intrinsics)
            points_3d.append(points_3d_frame)
        
        return torch.stack(points_3d)  # (2, N, 3)
    
    def _rescale_coordinates(self, points_2d: torch.Tensor) -> torch.Tensor:
        """Rescale 2D coordinates from VGG-SfM output to image coordinates."""
        points_2d_rescaled = points_2d.clone()
        points_2d_rescaled[:, 0] = points_2d_rescaled[:, 0] * self.x_scale
        points_2d_rescaled[:, 1] = points_2d_rescaled[:, 1] * self.y_scale + self.y_shift
        return points_2d_rescaled
    
    def _sample_depths(self, points_2d: torch.Tensor, depth_image: torch.Tensor) -> torch.Tensor:
        """Sample depth values at 2D point locations using bilinear interpolation."""
        h, w = depth_image.shape
        
        # Convert to normalized coordinates for grid_sample [-1, 1]
        points_2d_norm = points_2d.clone()
        points_2d_norm[:, 0] = 2 * points_2d_norm[:, 0] / (w - 1) - 1
        points_2d_norm[:, 1] = 2 * points_2d_norm[:, 1] / (h - 1) - 1
        points_2d_norm = points_2d_norm.view(1, 1, -1, 2)
        
        # Sample depths
        depths = torch.nn.functional.grid_sample(
            depth_image.unsqueeze(0).unsqueeze(0),
            points_2d_norm,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze()
        
        return depths
    
    def _unproject_points(self, 
                         points_2d: torch.Tensor, 
                         depths: torch.Tensor, 
                         intrinsics: torch.Tensor) -> torch.Tensor:
        """Unproject 2D points to 3D using depths and camera intrinsics."""
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        x = (points_2d[:, 0] - cx) * depths / fx
        y = (points_2d[:, 1] - cy) * depths / fy
        z = depths
        
        return torch.stack([x, y, z], dim=1)
    
    def _filter_valid_points(self, points_3d: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Filter out points with invalid depth values."""
        valid_mask = (points_3d[0, :, 2] > 0) & (points_3d[1, :, 2] > 0)
        points_3d_filtered = points_3d[:, valid_mask]
        
        source_points = points_3d_filtered[0].cpu().numpy()
        target_points = points_3d_filtered[1].cpu().numpy()
        
        return source_points, target_points
    
    def _ransac_pose_estimation(self, 
                               source_points: np.ndarray, 
                               target_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate pose using RANSAC with Arun's method."""
        best_transform = None
        best_inlier_count = 0
        best_inlier_mask = None
        
        for _ in range(self.ransac_iterations):
            # Random sample
            sample_indices = np.random.choice(len(source_points), self.min_points, replace=False)
            sample_source = source_points[sample_indices]
            sample_target = target_points[sample_indices]
            
            # Estimate transform using Arun's method
            transform = self._arun_method(sample_source, sample_target)
            
            # Evaluate on all points
            inlier_mask = self._evaluate_transform(transform, source_points, target_points)
            inlier_count = np.sum(inlier_mask)
            
            # Update best transform
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_transform = transform
                best_inlier_mask = inlier_mask
        
        # Refine using all inliers if enough found
        if best_inlier_count >= self.min_inlier_ratio * len(source_points):
            inlier_source = source_points[best_inlier_mask]
            inlier_target = target_points[best_inlier_mask]
            best_transform = self._arun_method(inlier_source, inlier_target)
        
        return best_transform, best_inlier_mask
    
    def _arun_method(self, source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
        """Compute transformation matrix using Arun's method."""
        # Center the points
        source_centered = source_points - source_points.mean(axis=0)
        target_centered = target_points - target_points.mean(axis=0)
        
        # Compute covariance matrix
        H = source_centered.T @ target_centered
        
        # SVD decomposition
        U, S, Vt = np.linalg.svd(H)
        
        # Compute rotation matrix
        R = Vt.T @ U.T
        
        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        
        # Compute translation
        t = target_points.mean(axis=0) - R @ source_points.mean(axis=0)
        
        # Create transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t
        
        return transform
    
    def _evaluate_transform(self, 
                           transform: np.ndarray, 
                           source_points: np.ndarray, 
                           target_points: np.ndarray) -> np.ndarray:
        """Evaluate transformation and return inlier mask."""
        R = transform[:3, :3]
        t = transform[:3, 3]
        
        transformed_points = (R @ source_points.T).T + t
        errors = np.linalg.norm(transformed_points - target_points, axis=1)
        
        return errors < self.inlier_threshold
    
    def _compute_covariance(self, 
                           source_points: np.ndarray, 
                           target_points: np.ndarray, 
                           transform: np.ndarray, 
                           inlier_mask: np.ndarray) -> np.ndarray:
        """Compute covariance matrix using numerical Jacobian."""
        inlier_source = source_points[inlier_mask]
        inlier_target = target_points[inlier_mask]
        num_points = len(inlier_source)
        
        if num_points < self.min_points:
            # Return large covariance if insufficient points
            return np.eye(6) * 1e6
        
        # Convert transformation to se(3) parameters
        R = transform[:3, :3]
        t = transform[:3, 3]
        axis_angle = Rotation.from_matrix(R).as_rotvec()
        current_params = np.concatenate([axis_angle, t])
        
        # Compute initial residuals
        transformed_points_current = (R @ inlier_source.T).T + t
        init_residuals = (inlier_target - transformed_points_current).reshape(-1)
        
        # Numerical Jacobian computation
        jacobian = np.zeros((num_points * 3, 6))
        
        for i in range(6):
            # Perturb parameter
            delta = np.zeros(6)
            delta[i] = self.numerical_eps
            perturbed_params = current_params + delta
            
            # Convert back to transformation
            perturbed_axis_angle = perturbed_params[:3]
            perturbed_t = perturbed_params[3:]
            perturbed_R = Rotation.from_rotvec(perturbed_axis_angle).as_matrix()
            
            # Compute perturbed residuals
            transformed_points_perturbed = (perturbed_R @ inlier_source.T).T + perturbed_t
            perturbed_residuals = (inlier_target - transformed_points_perturbed).reshape(-1)
            
            # Numerical derivative
            jacobian[:, i] = (perturbed_residuals - init_residuals) / self.numerical_eps
        
        # Compute covariance
        sigma = np.sqrt(np.mean(init_residuals**2)) * 1000.0
        try:
            covariance = sigma * np.linalg.inv(jacobian.T @ jacobian)
        except np.linalg.LinAlgError:
            # Handle singular matrix
            covariance = np.eye(6) * 1e6
        
        return covariance


def estimate_relative_pose(tracks: np.ndarray, 
                          intrinsics: np.ndarray, 
                          depth_images: np.ndarray,
                          **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function for relative pose estimation.
    
    Args:
        tracks: Feature tracks of shape (2, N, 2)
        intrinsics: Camera intrinsic matrix of shape (3, 3)
        depth_images: Depth images of shape (2, H, W)
        **kwargs: Additional parameters for RelativePoseEstimator
        
    Returns:
        Tuple of (transformation_matrix, covariance_matrix)
    """
    estimator = RelativePoseEstimator(**kwargs)
    return estimator.estimate_pose(tracks, intrinsics, depth_images)


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--H", type=int, default=480)
    args.add_argument("--W", type=int, default=640)
    args.add_argument("--N", type=int, default=2048)
    args.add_argument("--example", type=bool, default=True)
    args.add_argument("--data_dir", type=str, default="data")
    args.add_argument("--method", type=str, default="convenience", choices=["convenience", "class"])
    args.add_argument("--intrinsics", type=str, default="intrinsics.txt", help="fx, fy, cx, cy")
    args = args.parse_args()
    H, W, N = args.H, args.W, args.N
    # cx, cy, fx, fy = np.loadtxt(args.intrinsics)
    
    if args.example:
    # Create example data (replace with your actual data)
        tracks = np.random.rand(2, N, 2) * 1024  # VGG-SfM scaled coordinates, [frame， num_points，xy_coordinates]
        fx, fy, cx, cy = 525.0, 525.0, 320.0, 240.0
        depth_images = np.random.rand(2, H, W) * 5.0  # Depth in meters
    else:
        data_dir = args.data_dir
        tracks = np.loadtxt(Path(data_dir) / "tracks.txt")
        intrinsics = np.loadtxt(Path(data_dir) / "intrinsics.txt")
        depth_images = np.load(Path(data_dir) / "depth_images.npy") # [frame, H, W]
        
    intrinsics = np.array([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1.0]])
        
    # Method 1: Using the convenience function
    if args.method == "convenience":
        transform, covariance = estimate_relative_pose(tracks, intrinsics, depth_images)
    elif args.method == "class":
    # Method 2: Using the class directly for more control
        estimator = RelativePoseEstimator(
            ransac_iterations=200,
            inlier_threshold=0.05,
            min_inlier_ratio=0.6,
            H=H,
            W=W
            )
        transform, covariance = estimator.estimate_pose(tracks, intrinsics, depth_images)
    else:
        raise ValueError(f"Invalid method: {args.method}")

    print(f"Covariance matrix shape: {covariance.shape}")