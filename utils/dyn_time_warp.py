import argparse

# Argument parser for command line execution
parser = argparse.ArgumentParser(description='Execute combined_overlay_function with specified exercise.')
parser.add_argument('--exercise', type=str, required=True, help='Name of the exercise to process.')

args = parser.parse_args()


import numpy as np
import cv2
import os
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def calculate_angle(point1, point2, point3):
    vector1 = np.array(point1) - np.array(point2)
    vector3 = np.array(point3) - np.array(point2)

    dot_product = np.dot(vector1, vector3)
    norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector3)

    cosine_angle = dot_product / norm_product
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

# Dictionary of angle
def name_angle_fun():
    name_angle1 = {'right_elbow': 0, 'left_elbow': 1, 'right_shoulder': 2, 'left_shoulder': 3,
              'right_knee': 4, 'left_knee': 5, 'right_hip':6, 'left_hip': 7, 'vertical': 8}
    return name_angle1

# Các điểm tính góc
def angle_dict_fun():
    angle_dict1 = {'right_elbow': [14,15,16], 'left_elbow': [11,12,13], 'right_shoulder': [8,14,15], 'left_shoulder': [8,11,12],
              'right_knee': [1,2,3], 'left_knee': [4,5,6], 'right_hip':[0,1,2], 'left_hip': [0,4,5],  'vertical': [8,0,0]}
    return angle_dict1

"""EXTRACT VIDEO"""


name_angle = name_angle_fun()
angle_dict = angle_dict_fun()
def extract_vid(array):
    matrix = []
    for i in range(array.shape[0]):  # calculate angle
        theta = []
        for key in angle_dict.keys():
            if key != "vertical":
                val = angle_dict[key]
                a = array[i][val[0]]
                b = array[i][val[1]]
                c = array[i][val[2]]
            else:
                val = angle_dict[key]
                a = array[i][val[0]]
                b = array[i][val[1]]
                c = [0,0,1]
            angle = calculate_angle(a, b, c)
            theta.append(angle)
        matrix.append(theta)
    return np.array(matrix)

"""CALCULATE DISTANCE MATRIX"""



#Input: 2 matrix are represented for 2 videos
def distance_matrix(mat1, mat2):
    N = mat1.shape[0]
    M = mat2.shape[0]
    dist_mat = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            dist_mat[i, j] = np.linalg.norm(mat1[i, :] - mat2[j,:] )
    return dist_mat

"""DYNAMIC TIME WARPING ALGORITHM"""




# Input: a distance matrix in which each element is distance of 2 vectors represented for 2 frames
# Output: list contains pairs of frame indexes of 2 videos
def dtw(dist_mat):
    N, M = dist_mat.shape
    cost_mat = np.zeros((N + 1, M + 1))
    for i in range(1,N+1):
        cost_mat[i, 0]  = np.inf
    for j in range(1, M+1):
        cost_mat[0, j] = np.inf
    traceback_mat = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            min_list = [cost_mat[i, j], # match = 0
                        cost_mat[i, j+1],   #insert = 1
                        cost_mat[i+1, j]]   # deletion = 2
            index_min = np.argmin(min_list)
            cost_mat[i+1,j+1] = dist_mat[i, j] + min_list[index_min]
            traceback_mat[i,j] = index_min
    i = N-1
    j = M -1
    path = [(i,j)]
    while i > 0 or j > 0:
        tb_type = traceback_mat[i,j]
        if tb_type == 0: # đi chéo
            i = i-1
            j = j-1
        elif tb_type == 1: # đi xuống
            i = i - 1
        elif tb_type == 2: # đi ngang
            j = j - 1

        path.append((i,j))
    cost_mat = cost_mat[1:, 1:]
    return path[::-1]

# """GET PATH FROM DTW"""

# def get_path1(path):
#     # Create a defaultdict to group elements by their first element
#     x = path[-1]
#     if x[0] > x[1]:

#         grouped = defaultdict(list)
#         for element in path:
#             grouped[element[1]].append(element)

#         # Find the element e with the maximum second element for each group
#         result = [max(group, key=lambda x: x[0]) for group in grouped.values()]
#     else:
#         grouped = defaultdict(list)
#         for element in path:
#             grouped[element[0]].append(element)

#         # Find the element e with the maximum second element for each group
#         result = [max(group, key=lambda x: x[1]) for group in grouped.values()]

#     return result

""" VIDEO OUTPUT OF TRAINER VIDEO"""
def get_trainer_output_vid(trainer_point_path, learner_point_path,trainer_video_path, 
                           trainer_output_video_path):
    cap = cv2.VideoCapture(trainer_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(trainer_output_video_path, fourcc, 10.0, (int(cap.get(3)), int(cap.get(4))))


    array_t = get_array(trainer_point_path)
    array_l = get_array(learner_point_path)
    
    array_l = align_poses(array_t, array_l)
    
    
    mat1 = extract_vid(array_t)
    mat2 = extract_vid(array_l)
    dist_mat = distance_matrix(mat1, mat2)
    path_dtw = dtw(dist_mat)
    #path_dtw = get_path1(path)
    stop = 0
    while cap.isOpened():
        stop += 1
        if stop == 2:
            break
        for i in path_dtw:
            cap.set(1, i[0])  # Where frame_no is the frame you want
            ret, frame = cap.read()
            if ret:
                out.write(frame)
            else:
                break
        
    cap.release()
    out.release()


"""GET COORDINATE ARRAY FROM FILE.NPY"""


def get_array(path):
    b = np.load(path)
    return b

"""GET WRONG LIST ANGLE + FRAME INDEX"""


def get_wrong_angle_dict(inpath1, inpath2, path, mat1, mat2):
    cap1 = cv2.VideoCapture(inpath1)
    cap2 = cv2.VideoCapture(inpath2)
    stop = 0
    name_angle = name_angle_fun()
    wrong_angle_list = []
    while cap1.isOpened() or cap2.isOpened():

        stop += 1
        if stop == 2:
            break
        for i in path:
            ang_list = []
            for key in name_angle.keys():
                j = name_angle[key]
                wrong_ang = abs(mat2[i[1],j] - mat1[i[0],j])

                if wrong_ang >= 20:
                    ang_list.append(key)
            wrong_angle_list.append([i[1],ang_list])
    return dict(wrong_angle_list)


# overlay on original video

def draw_3d_keypoints_on_frame(frame, keypoints_3d, color_3d=(0, 255, 0)):
    """
    Draw 3D keypoints on a given frame.
    """
    for kp in keypoints_3d:
        x_3d, y_3d, _ = kp
        cv2.circle(frame, (int(x_3d), int(y_3d)), 3, color_3d, -1)
    return frame

def overlay_3d_on_original_video(original_video_path, keypoints_3d_path, output_video_path):
    # Load the 3D keypoints
    pose_3d_data = np.load(keypoints_3d_path)

    # Open the original video
    cap = cv2.VideoCapture(original_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 10.0, (int(cap.get(3)), int(cap.get(4))))

    # Check if video writer is opened successfully
    if not out.isOpened():
        print("Error: Video writer could not be opened!")
        return

    frame_idx = 0
    if not cap.isOpened():
      print("Error: Video capture could not be opened!")
      return
    else:
      print(f"Video capture opened. Total frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

    while cap.isOpened():
        print(f"Processing frame: {frame_idx}")
        ret, frame = cap.read()
        if ret:
            if frame_idx < pose_3d_data.shape[0]:
                keypoints_3d = pose_3d_data[frame_idx]
                frame = draw_3d_keypoints_on_frame(frame, keypoints_3d)
                out.write(frame)
                frame_idx += 1
            else:
                print(f"Processed all keypoints. Last frame index: {frame_idx}")
                break
        else:
            print(f"Video read returned False at frame index: {frame_idx}")
            break

    cap.release()
    out.release()

    print(f"Overlay completed. The output video is saved at: {output_video_path}")


def highlight_body_part_with_circle(frame, keypoints, body_parts, circle_radius=10):
    """
    Highlight the specified body parts using circles around the keypoints with side-specific colors.
    """
    # Define the keypoints indices for various body parts
    angle_dict = {
        'right_elbow': [14, 15, 16],
        'left_elbow': [11, 12, 13],
        'right_shoulder': [8, 14, 15],
        'left_shoulder': [8, 11, 12],
        'right_knee': [1, 2, 3],
        'left_knee': [4, 5, 6],
        'right_hip': [0, 1, 2],
        'left_hip': [0, 4, 5],
        'vertical': [8, 0, 0]
    }

    # Colors for left and right side keypoints
    color_dict = {
        'left': (0, 0, 139),  # Blue for left side keypoints
        'right': (0, 0, 255)  # Red for right side keypoints
    }

    # For each body part in the list, extract the middle keypoint and highlight it with the appropriate color
    for body_part in body_parts:
        middle_idx = angle_dict[body_part][1]
        kp = keypoints[middle_idx][:2]

        # Determine color based on left or right
        if "left" in body_part:
            color = color_dict["left"]
        else:
            color = color_dict["right"]

        cv2.circle(frame, (int(kp[0]), int(kp[1])), circle_radius, color, -1)

    return frame

def combined_overlay_function(trainer_point_path, learner_point_path, trainer_video_path, learner_video_path
                              , output_video_path):
    # Load the 3D keypoints
    
    array_t = get_array(trainer_point_path)
    array_l = get_array(learner_point_path)
    
    array_t = align_poses(array_l, array_t)
    
    mat1 = extract_vid(array_t)
    mat2 = extract_vid(array_l)
    dist_mat = distance_matrix(mat1, mat2)
    path = dtw(dist_mat)
    path_dtw = get_path1(path)
    path_learner = [i[1] for i in path_dtw]
    
    #wrong_angle = get_wrong_angle_dict(trainer_video_path, learner_video_path, path_dtw, mat1, mat2)
    
    body_parts_to_highlight = get_wrong_angle_dict(trainer_video_path, learner_video_path, path_dtw, mat1, mat2)
    pose_3d_data = array_l

    # Open the original video
    cap = cv2.VideoCapture(learner_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 10.0, (int(cap.get(3)), int(cap.get(4))))

    # Check if video writer is opened successfully
    if not out.isOpened():
        print("Error: Video writer could not be opened!")
        return

    frame_idx = 0
    if not cap.isOpened():
        print("Error: Video capture could not be opened!")
        return
    else:
        print(f"Video capture opened. Total frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if frame_idx in path_learner:
                keypoints_3d = pose_3d_data[frame_idx]

                # Draw all 3D keypoints on the frame
                frame = draw_3d_keypoints_on_frame(frame, keypoints_3d)

                # Highlight specified body parts
                if frame_idx in body_parts_to_highlight.keys():
                    frame = highlight_body_part_with_circle(frame, keypoints_3d, body_parts_to_highlight[frame_idx], circle_radius=15)

                out.write(frame)
                frame_idx += 1
            else:
                pass
        else:
            break

    cap.release()
    out.release()

    print(f"Overlay completed. The output video is saved at: {output_video_path}")

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

# File paths based on the exercise argument
EXERCISE = args.exercise
trainer_point_path = f"/data/trainer/{EXERCISE}/X3D.npy"
learner_point_path = f"/data/learner/{EXERCISE}/X3D.npy"
trainer_video_path = f"/data/trainer/{EXERCISE}/{EXERCISE}.mp4"
learner_video_path = f"/data/learner/{EXERCISE}/{EXERCISE}.mp4"
learner_output_video_path = f"/data/learner/{EXERCISE}/output.mp4"
trainer_output_video_path = f"/data/trainer/{EXERCISE}/output.mp4"

# Call the combined function if the script is executed directly
if __name__ == '__main__':
    combined_overlay_function(trainer_point_path, learner_point_path,trainer_video_path, 
                              learner_video_path, learner_output_video_path)

    get_trainer_output_vid(trainer_point_path, learner_point_path,trainer_video_path, 
                           trainer_output_video_path)
