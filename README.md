# MotionMentor: 'Learn by Doing' with 3D Human Pose Estimation and Dynamic Time Warping for Accurate Pose Comparison in Fitness
## Result 
The Human Pose Feedback System application is versatile and can be adapted for various types of workouts, including martial arts, hypertrophy training, and dance choreography. 

1. **Martial Arts**: The application can analyze and provide feedback on posture, alignment, and movement precision crucial in martial arts. It helps in refining stances, kicks, and strikes through detailed pose analysis. <br>
   ### Taekwondon <br>

   ![taek (2)](https://github.com/vuxminhan/Human-pose-feedback-system/assets/54212949/83ede9c0-ff13-480d-8ca8-61cc6a6aa7b7)

   ### Vovinam

   ![vovinam (2)](https://github.com/vuxminhan/Human-pose-feedback-system/assets/54212949/fb26c796-2f22-4973-ae35-fa12e607c85b)

   
2. **Hypertrophy Training**: For bodybuilding and strength training, the system can monitor and correct form during exercises. This aids in targeting specific muscle groups effectively and reducing the risk of injury.
   ### Air squat
   ![Air squat](https://github.com/vuxminhan/Human-pose-feedback-system/assets/54212949/187ae504-44b3-458c-a033-99c5bd4bd971)

   ### Front squat
   ![front_squat](https://github.com/vuxminhan/Human-pose-feedback-system/assets/54212949/84e177a9-b9b7-4c33-82da-49cb204650be)

4. **Dance Choreography**: The application offers valuable insights into posture and movement fluidity, essential for dancers. It can be used to refine and synchronize complex choreographic sequences, ensuring each movement is executed with precision.

   ### Kpop dance

   ![kpop_dance](https://github.com/vuxminhan/Human-pose-feedback-system/assets/54212949/2bfc75da-9ccf-42e1-a8b8-d99e8208557b)
   
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



