import argparse
import numpy as np
import time
from collections import defaultdict
import json

class Preprocessing:
    '''Preprocessing video...'''

    def __init__(self, file_directory_trainer = '', file_directory_learner = ''):
        '''Initialize the class with the file directory of the trainer and the learner.'''

        self.trainer_data = self.read_file(file_directory_trainer)
        self.learner_data = self.read_file(file_directory_learner)
    
    def read_file(self, file_directory):
        '''Read the file from the file directory and return the data.'''

        data = np.load(file_directory)
        return data
    
    def calculate_angle(self, point1, point2, point3):
        '''Calculate the angle between 3 points.
        Parameters
        ----------
        point1, point2, point3 : list
            The 3D- coordinates of the 3 points.
        '''

        vector1 = np.array(point1) - np.array(point2)
        vector3 = np.array(point3) - np.array(point2)

        dot_product = np.dot(vector1, vector3)
        norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector3)

        cosine_angle = dot_product / norm_product
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)
    
    def angle_dict_fun(self):
        angle_dict = {
                    'right_elbow': [14,15,16], 
                    'left_elbow': [11,12,13], 
                    'right_shoulder': [8,14,15], 
                    'left_shoulder': [8,11,12],
                    'right_knee': [1,2,3], 
                    'left_knee': [4,5,6], 
                    'right_hip':[0,1,2], 
                    'left_hip': [0,4,5],  
                    'vertical': [8,0,0]
                     }
        return angle_dict
    
    def extract_vid(self, data):
        matrix = []
        angle_dict = self.angle_dict_fun()
        for i in range(data.shape[0]):  #
            theta = []
            for key in angle_dict.keys():                  
                if key != "vertical": 
                    val = angle_dict[key]
                    a = data[i][val[0]]
                    b = data[i][val[1]]
                    c = data[i][val[2]]
                else:
                    val = angle_dict[key]
                    a = data[i][val[0]]
                    b = data[i][val[1]]
                    c = [0,1,0] # coordinate of vertical vector
                angle = self.calculate_angle(a, b, c)
                theta.append(angle)               
            matrix.append(theta)    
        return np.array(matrix)
    
    def distance_matrix(self, mat1, mat2):
        N = mat1.shape[0]
        M = mat2.shape[0]
        dist_mat = np.zeros((N,M))
        for i in range(N):
            for j in range(M):
                dist_mat[i, j] = np.linalg.norm(mat1[i, :] - mat2[j,:] )
        return dist_mat
    
    def execute(self, verbose = True):
        '''Execute the preprocessing and return the distance matrix.'''
        if verbose:
            print("Start preprocessing...")
            start_time = time.time()
            print('-'*50)
        
        trainer_matrix = self.extract_vid(self.trainer_data)
        learner_matrix = self.extract_vid(self.learner_data)
        distance_matrix = self.distance_matrix(trainer_matrix, learner_matrix)

        if verbose:
            print("Finish preprocessing!")
            print(f"Eslaped time: {time.time() - start_time}")
            print('-'*50)
        return distance_matrix, trainer_matrix, learner_matrix
    
class Dynamic_Time_Warping:
    '''Dynamic Time Warping class for the data.'''

    def __init__(self, distance_matrix, trainer_data, learner_data):
        '''Initialize the class with the distance matrix.'''

        self.distance_matrix = distance_matrix
        self.trainer_data = trainer_data
        self.learner_data = learner_data
    
    def dynamic_time_warping_algorithm(self, dist_mat):
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
            if tb_type == 0: 
                i = i-1
                j = j-1
            elif tb_type == 1: 
                i = i - 1
            elif tb_type == 2: 
                j = j - 1
        
            path.append((i,j))
        cost_mat = cost_mat[1:, 1:]
        return path[::-1]

    def optimize_path(self, path):
        '''Optimize the path by removing the redundant frames.'''
        x = path[-1]
        if x[0] > x[1]:
            grouped = defaultdict(list)
            for element in path:
                grouped[element[1]].append(element)

            # Find the element e with the maximum second element for each group
            result = [max(group, key=lambda x: x[0]) for group in grouped.values()]
        else:
            grouped = defaultdict(list)
            for element in path:
                grouped[element[0]].append(element)

            # Find the element e with the maximum second element for each group
            result = [max(group, key=lambda x: x[1]) for group in grouped.values()]

        return result
    
    def execute(self, verbose = True):
        '''Execute the dynamic time warping and return the path.'''

        if verbose:
            print("Start dynamic time warping...")
            start_time = time.time()
            print('-'*50)

        path = self.dynamic_time_warping_algorithm(self.distance_matrix)
        opt_path = self.optimize_path(path)

        if verbose:
            print("Finish dynamic time warping!")
            print(f"Eslaped time: {time.time() - start_time}")
            print('-'*50)

        return opt_path


class Compare:
    '''Compare the 2 videos.'''
    def __init__(self, path_dtw, trainer_data, learner_data, trainer_matrix, learner_matrix, output_path, angle_threshold):
        '''Initialize the class with the path of the dynamic time warping.'''

        self.path_dtw = path_dtw
        self.trainer_data = trainer_data
        self.learner_data = learner_data  
        self.trainer_matrix = trainer_matrix
        self.learner_matrix = learner_matrix
        self.output_path = output_path
        self.angle_threshold = angle_threshold

    def return_matched_file(self, path_dtw, data, opt = 0):
            '''Return the matched in .npy file.
            Parameters: 
            ----------'
            path_dtw : list
                The path of the dynamic time warping.
            data : numpy array
                The data of the video.
            opt : int
                0 for trainer, 1 for learner.'''

            new_data = np.array([data[0]])
            for i in path_dtw:
                j = i[opt] 
                new_data = np.append(new_data,[data[j]], axis=0)
            if opt == 0:
                name = 'trainer'
            else:
                name = 'learner'

            np.save(self.output_path+name+'.npy',new_data[1:])
        
    def angle_dict_fun(self):
        angle_dict = {
                        'RightElbow': [14,15,16], 
                        'LeftElbow': [11,12,13], 
                        'RightShoulder': [8,14,15], 
                        'LeftShoulder': [8,11,12],
                        'RightKnee': [1,2,3], 
                        'LeftKnee': [4,5,6], 
                        'RightHip':[0,1,2], 
                        'LeftHip': [0,4,5],  
                        'Vertical': [8,0,0]
                     }
        return angle_dict
    
    def name_angle_fun(self):
        name_angle = {
                        'RightElbow': 0, 
                        'LeftElbow': 1, 
                        'RightShoulder': 2, 
                        'LeftShoulder': 3,
                        'RightKnee': 4, 
                        'LeftKnee': 5, 
                        'RightHip':6, 
                        'LeftHip': 7, 
                        'Vertical': 8
                      }
        return name_angle
        
    def get_wrong_angle_list(self, path_dtw, mat1, mat2):
        wrong_angle_list = []
        name_angle = self.name_angle_fun()
        for i in path_dtw:
            ang_list = []
            for key in name_angle.keys():
                j = name_angle[key]
                wrong_ang = abs(mat2[i[1],j] - mat1[i[0],j])

                if wrong_ang >= self.angle_threshold:
                    ang_list.append(key)
            wrong_angle_list.append(ang_list)
        return wrong_angle_list

    def return_error_file(self,path_dtw, learner_data):
        name_error_angles = self.get_wrong_angle_list(path_dtw, self.trainer_matrix, self.learner_matrix)
        angle_dict = self.angle_dict_fun()
        error_data = []
        index = 0
        for i in path_dtw:
            error_frame = [[np.nan, np.nan, np.nan]] *17 
            key_list = name_error_angles[index]       
            if key_list:
                index_list = []
                for key in key_list:
                    j = angle_dict[key][1]
                    index_list.append(j)
                for k in range(1,17):
                    if k not in index_list:
                        learner_data[i[1],k] = [np.nan, np.nan, np.nan]
                        error_frame = learner_data[i[1]].tolist()
            error_data.append(error_frame)
            index += 1
        my_3d_array = np.array(error_data).reshape((len(error_data), 17, 3))    
        np.save(self.output_path + 'angles_error.npy',my_3d_array)
        with open( self.output_path + 'name_error_angles.json', 'w') as file:
                json.dump(name_error_angles, file) 

    def execute(self, verbose = True):
        '''Execute the dynamic time warping and return the path.'''
        if verbose:
            print("Start comparing 2 videos...")
            start_time = time.time()
            print('-'*50)

        self.return_matched_file(self.path_dtw, self.trainer_data, opt = 0)
        self.return_matched_file(self.path_dtw, self.learner_data, opt = 1)
        self.return_error_file(self.path_dtw, self.learner_data)

        if verbose:
            print("Finish comparing 2 videos!")
            print(f"Eslaped time: {time.time() - start_time}")
            print('-'*50)

def main():
    # Argument parser for command line execution
    parser = argparse.ArgumentParser(description='Execute combined_overlay_function with specified exercise.')
    parser.add_argument('--trainer_path', type=str, required=True, help='Path to the trainer data file.')
    parser.add_argument('--learner_path', type=str, required=True, help='Path to the learner data file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output files.')
    parser.add_argument('--angle_threshold', type=float, required=True, help='Angle difference in degree to be considered an error')

    args = parser.parse_args()

    trainer_path = args.trainer_path
    learner_path = args.learner_path
    output_path = args.output_path
    angle_threshold = args.angle_threshold

    distance_matrix, trainer_matrix, learner_matrix = Preprocessing(file_directory_trainer=trainer_path,
                                                                    file_directory_learner=learner_path).execute()

    trainer_data = np.load(trainer_path)
    learner_data = np.load(learner_path)

    path_dtw = Dynamic_Time_Warping(distance_matrix, trainer_data, learner_data).execute()

    Compare(path_dtw, trainer_data, learner_data, trainer_matrix, learner_matrix, output_path, angle_threshold).execute()


if __name__ == '__main__':
    main()

    
