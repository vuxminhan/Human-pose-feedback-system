import numpy as np
import argparse
from pathlib import Path
from pose2bvh.bvh_skeleton import h36m_skeleton

def main(input_npy, output_folder):
    # Extract base name from input file and create output BVH file name
    base_name = Path(input_npy).stem
    output_bvh = Path(output_folder) / f"{base_name}.bvh"
    output_npy = Path(output_folder) / "transformedX3D.npy"
    # Load the pose data
    pose3d_world = np.load(input_npy)
    print(f"Loaded pose data from {input_npy}")

    # Initialize the skeleton
    h36m_skel = h36m_skeleton.H36mSkeleton()
    print("Initialized H36mSkeleton")

    # Convert poses to BVH
    channels, header = h36m_skel.poses2bvh(pose3d_world, output_file=output_bvh)
    reshaped_channels = [channels[i:i+3] for i in range(0, len(channels), 3)]

    np.save(output_npy, reshaped_channels)
    print(f"Channels data saved to {output_folder}")
    print(f"Converted to BVH and saved to {output_bvh}")

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Convert 3D pose data to BVH format.")
    parser.add_argument("input_npy", help="Input numpy file path.")
    parser.add_argument("output_folder", help="Output folder for the BVH file.")
    args = parser.parse_args()

    # Run the main function
    main(args.input_npy, args.output_folder)
