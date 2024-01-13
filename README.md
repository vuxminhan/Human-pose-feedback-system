# MotionMentor: 'Learn by Doing' with 3D Human Pose Estimation and Dynamic Time Warping for Accurate Pose Comparison in Fitness
## Result 
The Human Pose Feedback System application is versatile and can be adapted for various types of workouts, including martial arts, hypertrophy training, and dance choreography. 
   
   ![vovinam (3)](https://github.com/vuxminhan/Human-pose-feedback-system/assets/54212949/1e108d34-d839-43bd-a531-5fdb62de6464)

1. **Hypertrophy Training**: For bodybuilding and strength training, the system can monitor and correct form during exercises. This aids in targeting specific muscle groups effectively and reducing the risk of injury.
   ### Air squat
   ![Air squat](https://github.com/vuxminhan/Human-pose-feedback-system/assets/54212949/187ae504-44b3-458c-a033-99c5bd4bd971)

   ### Front squat
   ![front_squat](https://github.com/vuxminhan/Human-pose-feedback-system/assets/54212949/84e177a9-b9b7-4c33-82da-49cb204650be)

2. **Dance Choreography**: The application offers valuable insights into posture and movement fluidity, essential for dancers. It can be used to refine and synchronize complex choreographic sequences, ensuring each movement is executed with precision.

   ### Kpop dance
   
   ![dance](https://github.com/vuxminhan/Human-pose-feedback-system/assets/54212949/2af503e9-6ac2-4034-b063-da1900efdae4)
3. **Martial Arts**: The application can analyze and provide feedback on posture, alignment, and movement precision crucial in martial arts. It helps in refining stances, kicks, and strikes through detailed pose analysis. <br>
   ### Taekwondon <br>

   ![taek (3)](https://github.com/vuxminhan/Human-pose-feedback-system/assets/54212949/dfa72be4-1a14-4060-afc7-29ea009fa516)

   ### Vovinam
## Installation

```bash
conda create -n pose-feedback python=3.7 anaconda
conda activate pose-feedback
# Please install PyTorch according to your CUDA version.
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```


## Applications

### In-the-wild inference (for custom videos)

Please refer to [infer_wild.ipynb](infer_wild.ipynb).



