import gtsam
import argparse
import numpy as np
from typing import List, Optional
from pathlib import Path


class PoseGraphOptimizer:
    """
    Pose Graph Optimization using GTSAM library.
    
    This class implements pose graph optimization for SLAM applications,
    combining absolute pose priors with relative pose constraints.
    """
    
    def __init__(self, num_frames: int):
        """
        Initialize the pose graph optimizer.
        
        Args:
            num_frames: Number of camera frames to optimize
        """
        self.num_frames = num_frames
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        
        # Initialize data structures
        self.absolute_estimates = [np.eye(4)] * num_frames
        self.absolute_pose_priors = [1e-5] + [1e-3] * (num_frames - 1)
        self.unreliable_frame_ids = []
        self.relative_poses = [np.eye(4)] * (num_frames - 1)
        self.relative_covariances = [np.eye(6)] * (num_frames - 1)
    
    def set_absolute_poses(self, poses: List[np.ndarray], priors: List[float]):
        """
        Set absolute pose estimates and their prior weights.
        
        Args:
            poses: List of 4x4 transformation matrices (camera from object)
            priors: List of prior weights (use low values for reliable frames)
        """
        assert len(poses) == self.num_frames, "Number of poses must match num_frames"
        assert len(priors) == self.num_frames, "Number of priors must match num_frames"
        
        self.absolute_estimates = poses
        self.absolute_pose_priors = priors
    
    def set_relative_poses(self, poses: List[np.ndarray], covariances: List[np.ndarray]):
        """
        Set relative pose estimates and their covariances.
        
        Args:
            poses: List of 4x4 relative transformation matrices (camera_i from camera_j)
            covariances: List of 6x6 covariance matrices for relative poses
        """
        assert len(poses) == self.num_frames - 1, "Need num_frames-1 relative poses"
        assert len(covariances) == self.num_frames - 1, "Need num_frames-1 covariances"
        
        self.relative_poses = poses
        self.relative_covariances = covariances
    
    def set_unreliable_frames(self, frame_ids: List[int]):
        """
        Mark frames as unreliable (will use large prior weights).
        
        Args:
            frame_ids: List of frame indices to mark as unreliable
        """
        self.unreliable_frame_ids = frame_ids
    
    def _add_absolute_pose_factors(self):
        """Add absolute pose prior factors to the graph."""
        for i, pose_matrix in enumerate(self.absolute_estimates):
            pose = gtsam.Pose3(pose_matrix)
            
            # Set prior noise based on reliability
            if i not in self.unreliable_frame_ids:
                prior_weight = self.absolute_pose_priors[i]
            else:
                prior_weight = 1.0  # Large weight for unreliable frames
            
            prior_noise = np.ones(6) * prior_weight
            noise_model = gtsam.noiseModel.Gaussian.Covariance(np.diag(prior_noise))
            
            # Add prior factor and initial estimate
            self.graph.add(gtsam.PriorFactorPose3(i, pose, noise_model))
            self.initial_estimate.insert(i, pose)
    
    def _add_relative_pose_factors(self):
        """Add relative pose constraint factors to the graph."""
        for i, (pose_matrix, cov_matrix) in enumerate(
            zip(self.relative_poses, self.relative_covariances)
        ):
            relative_pose = gtsam.Pose3(pose_matrix)
            
            # Decouple translation and rotation covariances
            cov = np.copy(cov_matrix)
            cov[:3, 3:] = 0
            cov[3:, :3] = 0
            
            noise_model = gtsam.noiseModel.Gaussian.Covariance(cov)
            self.graph.add(gtsam.BetweenFactorPose3(i, i + 1, relative_pose, noise_model))
    
    def optimize(self) -> List[np.ndarray]:
        """
        Perform pose graph optimization.
        
        Returns:
            List of optimized 4x4 pose matrices
        """
        # Clear previous graph and estimates
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        
        # Add factors to the graph
        self._add_absolute_pose_factors()
        self._add_relative_pose_factors()
        
        # Optimize
        parameters = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(
            self.graph, self.initial_estimate, parameters
        )
        optimized_values = optimizer.optimize()
        
        # Extract optimized poses
        optimized_poses = []
        for i in range(self.num_frames):
            optimized_poses.append(optimized_values.atPose3(i).matrix())
        
        return optimized_poses


def example_usage(args, weight_first_frame, weight_other_frames):
    """
    Example usage of the PoseGraphOptimizer.
    
    Usage instructions:
    1. Create optimizer instance, specifying number of frames
    2. Set absolute pose estimates and prior weights 
    (use small weight like 1e-5 for first frame, 1e-3 for other frames)
    3. Set relative pose estimates and covariance matrices
    4. Mark unreliable frames (optional)
    5. Call optimize() method to perform optimization
    """
    # Initialize optimizer
    if args.use_example:
        num_frames = 100
        optimizer = PoseGraphOptimizer(num_frames)
        
        # Set absolute poses and priors
        # substitute the absolute poses and priors with your own
        absolute_poses = [np.eye(4)] * num_frames
        
        # Set relative poses and covariances
        # substitute the relative poses and covariances with your own 
        # (from rel_pose_postproc.py)
        relative_poses = [np.eye(4)] * (num_frames - 1)
        relative_covariances = [np.eye(6)] * (num_frames - 1)
    else:
        data_dir = args.data_dir
        absolute_poses = np.loadtxt(Path(data_dir) / "absolute_poses.txt")
        relative_poses = np.loadtxt(Path(data_dir) / "relative_poses.txt")
        relative_covariances = np.loadtxt(Path(data_dir) / "relative_covariances.txt")
    
    absolute_priors = [weight_first_frame] + [weight_other_frames] * (num_frames - 1)
    optimizer.set_absolute_poses(absolute_poses, absolute_priors)
    optimizer.set_relative_poses(relative_poses, relative_covariances)
    
    # Mark unreliable frames (optional)
    unreliable_frames = []  # Add frame indices that should be down-weighted
    optimizer.set_unreliable_frames(unreliable_frames)
    
    # Optimize
    optimized_poses = optimizer.optimize()
    
    return optimized_poses


if __name__ == "__main__":
    # Run example
    args = argparse.ArgumentParser()
    args.add_argument("--data_dir", type=str, default="data")
    args.add_argument("--use_example", type=bool, default=True)
    args = args.parse_args()
    
    weight_1st_frame = 1e-5
    weight_other_frames = 1e-3
    
    optimized_poses = example_usage(args, weight_1st_frame, weight_other_frames)

    print(optimized_poses)
    print(f"Optimized {len(optimized_poses)} poses")
