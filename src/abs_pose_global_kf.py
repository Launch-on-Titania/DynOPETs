import argparse
import numpy as np
from pathlib import Path
from rich import print
from scipy.linalg import logm

from tools import (
    plot_three_pairs_comparison_subplots,
    load_tum_poses,
    save_tum_poses,
)


def unwrap_angles(angles):
    """Unwrap angles to avoid discontinuities."""
    return np.unwrap(angles, axis=0)


def rotation_matrix_to_euler_angles(R):
    """Convert rotation matrix to Euler angles (yaw, pitch, roll)."""
    assert R.shape == (3, 3), "Input rotation matrix must be 3x3"
    
    # Calculate pitch (beta)
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    # Calculate yaw (alpha)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    # Calculate roll (gamma)
    roll = np.arctan2(R[2, 1], R[2, 2])
    
    return np.array([yaw, pitch, roll])


def skew_symmetric(omega):
    """Create skew-symmetric matrix from angular velocity vector."""
    v = omega / np.linalg.norm(omega)
    return np.array([
        [0, -v[2], v[1]], 
        [v[2], 0, -v[0]], 
        [-v[1], v[0], 0]
    ])


def angular_velocity_to_rotation_matrix(omega, dt):
    """Convert angular velocity to rotation matrix using Rodrigues' formula."""
    theta = np.linalg.norm(omega) * dt
    if theta == 0:
        return np.eye(3)
    
    K = skew_symmetric(omega)
    # Rodrigues' formula
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    return R


def calculate_angular_velocity(R1, R2, dt):
    """Calculate angular velocity between two rotation matrices."""
    R_rel = np.dot(R2, np.linalg.inv(R1))
    log_R_rel = logm(R_rel)
    omega_rel = np.array([log_R_rel[2, 1], log_R_rel[0, 2], log_R_rel[1, 0]])
    omega = omega_rel / dt
    return omega  # rad/s yaw, pitch, roll


def calculate_trans_velocity(T1, T2, dt):
    """Calculate translation velocity between two positions."""
    vel_rel = T2 - T1
    vel = vel_rel / dt
    return vel


def record_rotation_translation(files, dt):
    """Record angular and translation velocities from pose matrices."""
    mtxs = np.array(files)
    angular_velocities = []
    trans_velocities = []

    for i in range(1, len(mtxs)):
        R1 = mtxs[i - 1][:3, :3]  # previous frame
        R2 = mtxs[i][:3, :3]      # current frame
        T1 = mtxs[i - 1][:3, 3]
        T2 = mtxs[i][:3, 3]
        
        omega = calculate_angular_velocity(R1, R2, dt)
        vel = calculate_trans_velocity(T1, T2, dt)
        
        angular_velocities.append(omega)
        trans_velocities.append(vel)
    
    # Duplicate the last frame to keep the same length
    angular_velocities.append(angular_velocities[-1])
    trans_velocities.append(trans_velocities[-1])

    return mtxs, angular_velocities, trans_velocities


def rts_smoothing(filtered_states, filtered_covs, predicted_states, predicted_covs, F):
    """Rauch-Tung-Striebel smoothing algorithm."""
    n = len(filtered_states)
    smoothed_states = [None] * n
    smoothed_covs = [None] * n

    # Initialization: the smoothed result at the last time step is the filtered result
    smoothed_states[-1] = filtered_states[-1]
    smoothed_covs[-1] = filtered_covs[-1]

    for k in range(n-2, -1, -1):
        try:
            inv_pred_cov = np.linalg.inv(predicted_covs[k+1])
        except np.linalg.LinAlgError:  # singular matrix
            inv_pred_cov = np.linalg.pinv(predicted_covs[k+1])
        
        # Gain matrix
        G = filtered_covs[k] @ F.T @ inv_pred_cov
        
        # Smoothed state
        smoothed_states[k] = filtered_states[k] + \
            G @ (smoothed_states[k+1] - predicted_states[k+1])

        # Smoothed covariance
        smoothed_covs[k] = filtered_covs[k] + \
            G @ (smoothed_covs[k+1] - predicted_covs[k+1]) @ G.T

    return smoothed_states, smoothed_covs


def process_rotation_matrices(mtxs, angular_velocities, ekf_refine, use_rts=True):
    """Process rotation matrices using EKF and optional RTS smoothing."""
    # Forward filtering: store all intermediate results
    filtered_states = []
    filtered_covs = []
    predicted_states = []
    predicted_covs = []
    euler_raws = []
    euler_opts = []
    euler_global_opts = []

    for i, matrix in enumerate(mtxs):
        if i == 0:
            # Initial state
            state = mtxs[0][:3, :3].flatten()
            omega = angular_velocities[0]
            update_P = ekf_refine.P

            # Store initial values
            filtered_states.append(state.copy())
            filtered_covs.append(update_P.copy())
            predicted_states.append(state.copy())
            predicted_covs.append(update_P.copy())

            smoothed_4x4 = mtxs[0].copy()
        else:
            # Extract and flatten rotation matrix
            measurement = matrix[:3, :3].flatten()

            # Prediction step
            state, omega, predicted_P = ekf_refine.predict(
                state, omega, update_P)

            # Store prediction results
            predicted_states.append(state.copy())
            predicted_covs.append(predicted_P.copy())

            # Update step
            state, update_P = ekf_refine.update(
                state, measurement, predicted_P)

            # Store filtering results
            filtered_states.append(state.copy())
            filtered_covs.append(update_P.copy())

            # Build current frame transformation matrix
            smoothed_matrix = state.reshape((3, 3))
            smoothed_4x4 = np.eye(4)
            smoothed_4x4[:3, :3] = smoothed_matrix
            smoothed_4x4[:3, 3] = matrix[:3, 3]

        # Calculate Euler angles for analysis
        yaw_raw, pitch_raw, roll_raw = rotation_matrix_to_euler_angles(
            mtxs[i][:3, :3])
        yaw_opt, pitch_opt, roll_opt = rotation_matrix_to_euler_angles(
            smoothed_4x4[:3, :3]
        )

        euler_opts.append([yaw_opt, pitch_opt, roll_opt])
        euler_raws.append([yaw_raw, pitch_raw, roll_raw])
        euler_global_opts.append([yaw_raw, pitch_raw, roll_raw])

    if use_rts:
        # RTS smoothing
        smoothed_states, smoothed_covs = rts_smoothing(
            filtered_states, filtered_covs,
            predicted_states, predicted_covs,
            ekf_refine.F
        )
    else:
        smoothed_states = filtered_states
        smoothed_covs = filtered_covs

    # Reconstruct pose matrices from smoothed results
    final_smoothed_mtxs = []
    for i, state in enumerate(smoothed_states):
        smoothed_matrix = state.reshape((3, 3))
        smoothed_4x4 = np.eye(4)
        smoothed_4x4[:3, :3] = smoothed_matrix
        smoothed_4x4[:3, 3] = mtxs[i][:3, 3]  # Keep original translation
        final_smoothed_mtxs.append(smoothed_4x4)

        # Update smoothed Euler angles (visualize only)
        if i < len(euler_opts):
            yaw, pitch, roll = rotation_matrix_to_euler_angles(smoothed_matrix)
            euler_global_opts[i] = [yaw, pitch, roll]

    euler_opts = np.array(euler_opts)
    euler_raws = np.array(euler_raws)
    euler_global_opts = np.array(euler_global_opts)

    return final_smoothed_mtxs, euler_raws, euler_opts, euler_global_opts, smoothed_states, smoothed_covs


def process_trans_matrices(smoothed_rotations_mtxs, trans_velocities, ekf_translation, 
                          use_rts=True, USE_SMOOTHED_TRANSLATION=False):
    """Process translation matrices using EKF and optional RTS smoothing."""
    # Store all intermediate results for RTS smoothing
    filtered_states = []      # Filtered states (x, y, z)
    filtered_covs = []        # Filtered covariance matrices
    predicted_states = []     # Predicted states
    predicted_covs = []       # Predicted covariances

    smoothed_mtxs = []        # Final output pose matrices
    trans_raws = []           # Raw translations
    trans_opts = []           # Filtered translations
    trans_smoothed = []       # Smoothed translations

    for i, matrix in enumerate(smoothed_rotations_mtxs):
        if i == 0:
            # Initialize state (x, y, z)
            state = smoothed_rotations_mtxs[0][:3, 3].copy()
            vel = trans_velocities[0].copy()
            update_P = ekf_translation.P.copy()

            # Store initial values
            filtered_states.append(state.copy())
            filtered_covs.append(update_P.copy())
            predicted_states.append(state.copy())
            predicted_covs.append(update_P.copy())

            # Initial pose matrix
            smoothed_4x4 = matrix.copy()
        else:
            # Get current measurement (x, y, z)
            measurement = matrix[:3, 3].copy()

            # Prediction step
            state, vel, predict_P = ekf_translation.predict(
                state, vel, update_P)
            predicted_states.append(state.copy())
            predicted_covs.append(predict_P.copy())

            # Update step
            state, update_P = ekf_translation.update(
                state, measurement, predict_P)
            filtered_states.append(state.copy())
            filtered_covs.append(update_P.copy())

            # Build current frame pose matrix
            smoothed_4x4 = np.eye(4)
            # Use already smoothed rotation
            smoothed_4x4[:3, :3] = matrix[:3, :3]
            smoothed_4x4[:3, 3] = state            # Use filtered translation

        # Store results
        smoothed_mtxs.append(smoothed_4x4)
        trans_raws.append(matrix[:3, 3])
        trans_opts.append(state.copy())

    if use_rts:
        # RTS smoothing
        smoothed_states, smoothed_covs = rts_smoothing(
            filtered_states=filtered_states,
            filtered_covs=filtered_covs,
            predicted_states=predicted_states,
            predicted_covs=predicted_covs,
            F=ekf_translation.F  # Get state transition matrix from EKF
        )
    else:
        smoothed_states = filtered_states
        smoothed_covs = filtered_covs

    final_smoothed_mtxs = []
    
    for i, smoothed_state in enumerate(smoothed_states):
        if USE_SMOOTHED_TRANSLATION:
            smoothed_4x4 = np.eye(4)
            # Keep smoothed rotation
            smoothed_4x4[:3, :3] = smoothed_rotations_mtxs[i][:3, :3]
            # Use smoothed translation
            smoothed_4x4[:3, 3] = smoothed_state
        else:
            smoothed_4x4 = smoothed_rotations_mtxs[i]

        final_smoothed_mtxs.append(smoothed_4x4)

    return final_smoothed_mtxs, smoothed_states, smoothed_covs


class EKFRotationSmoothing:
    """Extended Kalman Filter for rotation matrix smoothing."""
    
    def __init__(self, fps):
        self.process_noise = 10
        self.measurement_noise = 150
        self.est_error = 1e-5

        self.dt = 1.0 / fps
        self.P = np.eye(9) * self.est_error
        # Process noise and measurement noise
        self.Q = np.eye(9) * self.process_noise
        self.R = np.eye(9) * self.measurement_noise

        self.F = self._compute_state_transition_matrix()

    def _compute_state_transition_matrix(self):
        """Compute the approximate state transition matrix for rotation.
        
        For second-order constant coefficient linear system for rotation matrix:
        - State vector: [R11, R12, R13, R21, R22, R23, R31, R32, R33]
        - Using second-order dynamics: x_{k+1} = 2*x_k - x_{k-1} + dt^2*u_k
        - Approximated as: F = I + dt*A where A represents angular velocity coupling
        
        Returns:
            np.ndarray: 9x9 state transition matrix
        """
        # Initialize identity matrix
        F = np.eye(9)
        
        # Time step parameters
        dt = self.dt
        dt2 = dt * dt
        
        # Add second-order dynamics coupling between rotation matrix elements
        # Each 3x3 block represents coupling between different rows of rotation matrix
        for i in range(3):
            for j in range(3):
                base_idx = i * 3 + j
                
                # Add velocity-like coupling terms for non-diagonal elements
                if i != j:
                    F[base_idx, base_idx] = 1 + dt2 * 0.1
                    
                    # Add cross-coupling between different elements
                    for k in range(3):
                        if k != j:
                            cross_idx = i * 3 + k
                            F[base_idx, cross_idx] += dt2 * 0.01
        
        return F

    def state_transition(self, state, omega):
        """State transition function for rotation."""
        R = state.reshape((3, 3))
        R_new = np.dot(R, angular_velocity_to_rotation_matrix(omega, self.dt))
        omega_new = calculate_angular_velocity(R, R_new, self.dt)
        state_new = R_new.copy().flatten()
        return state_new, omega_new

    def predict(self, state, omega, update_P):
        """Prediction step of EKF."""
        # Predict state
        predicted_state, omega_new = self.state_transition(state, omega)
        predicted_P = self.F @ update_P @ self.F.T + self.Q
        return predicted_state, omega_new, predicted_P

    def update(self, state, measurement, predicted_P):
        """Update step of EKF."""
        y = measurement - state  # Innovation
        S = predicted_P + self.R  # Innovation covariance
        K = np.dot(predicted_P, np.linalg.inv(S))  # Kalman gain
        updated_state = state + np.dot(K, y)
        updated_P = predicted_P - np.dot(K, predicted_P)
        return updated_state, updated_P


class EKFTranslationSmoothing:
    """Extended Kalman Filter for translation smoothing."""
    
    def __init__(self, fps):
        self.process_noise = 10
        self.measurement_noise = 100
        self.est_error = 1e-5

        self.dt = 1.0 / fps
        self.P = np.eye(3) * self.est_error
        self.Q = np.eye(3) * self.process_noise
        self.R = np.eye(3) * self.measurement_noise
        self.F = np.eye(3)

    def state_transition(self, state, vel):
        """State transition function for translation."""
        T_new = state + vel * self.dt
        return T_new, vel

    def predict(self, state, vel, update_P):
        """Prediction step of EKF."""
        predicted_state, vel_new = self.state_transition(state, vel)
        predicted_P = update_P + self.Q
        return predicted_state, vel_new, predicted_P

    def update(self, state, measurement, predicted_P):
        """Update step of EKF."""
        y = measurement - state
        S = predicted_P + self.R
        K = np.dot(predicted_P, np.linalg.inv(S))
        updated_state = state + np.dot(K, y)
        updated_P = predicted_P - np.dot(K, predicted_P)
        return updated_state, updated_P


def main():
    """Main function for pose smoothing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--seqs_id", type=str, default="bottle_00")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--vis_on_subplots", action="store_true")
    parser.add_argument("--log_dir", type=str, default="temp_test")
    args = parser.parse_args()
    
    USE_SMOOTHED_TRANSLATION = False
    base_dir = "~/Datasets/DynOPETs/COPE119"
    initial_abs_pose_dir = "temp_dir"
    raw_abs_pose_parent_dir = Path(base_dir).expanduser() / args.seqs_id 
    smoothed_pose_save_dir = Path(initial_abs_pose_dir).expanduser() / "Global_KF_Opt"
    
    if not smoothed_pose_save_dir.exists():
        smoothed_pose_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Data from FoundationPose or BundleSDF 
    data_dir = raw_abs_pose_parent_dir / f"{args.seqs_id}_o2c_pose.abspose.txt"
    tum_save_path = smoothed_pose_save_dir / f"{args.seqs_id}_o2c_pose.abspose.gkf.tum.txt"
    
    # Create log_dir if it does not exist
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)

    timestamps, poses = load_tum_poses(data_dir)
    dt = 1.0 / args.fps 

    # Initialize EKF instances
    ekf_rotation = EKFRotationSmoothing(args.fps)
    ekf_translation = EKFTranslationSmoothing(args.fps)

    mtxs, angular_velocities, trans_velocities = record_rotation_translation(
        poses, dt)

    # Process rotation matrices with RTS smoothing
    smoothed_rotations_mtxs, euler_raws, euler_opts, euler_global_opts, smoothed_states, smoothed_covs = process_rotation_matrices(
        mtxs, angular_velocities, ekf_rotation)
    
    # Process translation matrices
    smoothed_final_mtxs, smoothed_states, smoothed_covs = process_trans_matrices(
        smoothed_rotations_mtxs, trans_velocities, ekf_translation, 
        USE_SMOOTHED_TRANSLATION=USE_SMOOTHED_TRANSLATION)
    # TODO: save smoothed_states, smoothed_covs
    
    # Save results
    save_tum_poses(tum_save_path, timestamps, smoothed_final_mtxs)
    print(f"Smoothed TUM pose file saved to: {tum_save_path}")

    # Prepare data for visualization
    yaw_pairs = [
        unwrap_angles(euler_raws[:, 0]), 
        unwrap_angles(euler_opts[:, 0]), 
        unwrap_angles(euler_global_opts[:, 0])
    ]
    pitch_pairs = [
        unwrap_angles(euler_raws[:, 1]), 
        unwrap_angles(euler_opts[:, 1]), 
        unwrap_angles(euler_global_opts[:, 1])
    ]
    roll_pairs = [
        unwrap_angles(euler_raws[:, 2]), 
        unwrap_angles(euler_opts[:, 2]), 
        unwrap_angles(euler_global_opts[:, 2])
    ]
    
    plot_three_pairs_comparison_subplots(
        args.log_dir, args.seqs_id, yaw_pairs, pitch_pairs, roll_pairs, 
        show=args.vis_on_subplots
    )


if __name__ == "__main__":
    main()
