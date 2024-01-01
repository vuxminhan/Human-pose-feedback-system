# pose2bvh

pose2bvh estimates skeleton information from 3D pose, converts 3D pose to joint angle and write motion data to bvh file.

步骤：使用MotionBERT预测出h36m格式的3d pose(T,17,3)，然后使用脚本转换成bvh。
1. [xian'sh](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md#halpe-dataset-26-keypoints)https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md#halpe-dataset-26-keypoints， 使用脚本提取2d关键点
   ![image](https://github.com/TnoobT/pose2bvh/assets/44052395/39e14637-13f9-4ac9-9cd5-722b684a86e0)
2. 使用motionBERT里的脚本，推理得到3d pose
   ![image](https://github.com/TnoobT/pose2bvh/assets/44052395/cbb91783-ea76-450e-95cd-62b3c8b18d96)
3. 使用这个脚本转换bvh即可。
