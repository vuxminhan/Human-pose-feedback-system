import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R

def align_poses(pose1, pose2):
    # Calculate the average vectors for the neck to nose direction for both poses
    avg_vector_neck_nose_pose1 = np.mean(pose1[:, 8, :] - pose1[:, 9, :], axis=0)
    avg_vector_neck_nose_pose2 = np.mean(pose2[:, 8, :] - pose2[:, 9, :], axis=0)
    
    # Normalize the average vectors
    avg_vector_neck_nose_pose1_norm = avg_vector_neck_nose_pose1 / np.linalg.norm(avg_vector_neck_nose_pose1)
    avg_vector_neck_nose_pose2_norm = avg_vector_neck_nose_pose2 / np.linalg.norm(avg_vector_neck_nose_pose2)
    
    # Calculate the cross product and the angle between the average vectors
    cross_product_avg = np.cross(avg_vector_neck_nose_pose2_norm, avg_vector_neck_nose_pose1_norm)
    angle_avg = np.arccos(np.clip(np.dot(avg_vector_neck_nose_pose2_norm, avg_vector_neck_nose_pose1_norm), -1.0, 1.0))
    
    # Create the rotation vector (axis-angle representation) for the average vectors
    rotation_vector_avg = cross_product_avg * angle_avg
    
    # Convert the rotation vector to a rotation matrix
    rotation_avg = R.from_rotvec(rotation_vector_avg)
    
    # Apply the rotation to all points of pose 2
    pose2_aligned_avg = np.empty_like(pose2)
    for i in range(pose2.shape[0]):
        pose2_aligned_avg[i] = rotation_avg.apply(pose2[i])
    
    return pose2_aligned_avg

def main():
    parser = argparse.ArgumentParser(description="Align the body facing direction of the second pose to the first pose.")
    parser.add_argument("pose1_path", type=str, help="Path to the .npy file for pose 1")
    parser.add_argument("pose2_path", type=str, help="Path to the .npy file for pose 2")
    parser.add_argument("output_path", type=str, help="Path to save the aligned pose 2 data")
    args = parser.parse_args()

    # Load the pose data from the .npy files
    pose1 = np.load(args.pose1_path)
    pose2 = np.load(args.pose2_path)

    # Perform the alignment
    pose2_aligned = align_poses(pose1, pose2)

    # Save the aligned Pose 2 data to a new .npy file
    np.save(args.output_path, pose2_aligned)

    print(f"Aligned pose data saved to: {args.output_path}")

if __name__ == "__main__":
    main()

