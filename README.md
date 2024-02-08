# AI fitness Trainer using 3D Human Pose Estimation and Dynamic Time Warping
## Result 
The Human Pose Feedback System application is versatile and can be adapted for various types of workouts, including martial arts, hypertrophy training, and dance choreography. 
   


1. **Hypertrophy Training**: For bodybuilding and strength training, the system can monitor and correct form during exercises. This aids in targeting specific muscle groups effectively and reducing the risk of injury.
   ### Air squat <br>
   
   ![air_squat_gif](https://github.com/vuxminhan/Human-pose-feedback-system/assets/54212949/e83491cd-d867-4499-8a18-4766f919be35)

   ### Front squat <br>
   
   ![front_squat_gif](https://github.com/vuxminhan/Human-pose-feedback-system/assets/54212949/6c39a679-19bb-4bec-bd66-a35375d4108c)

3. **Dance Choreography**: The application offers valuable insights into posture and movement fluidity, even for fast-moving subjects like dancers. It can be used to refine and synchronize complex choreographic sequences, ensuring each movement is executed with precision.

   ### Kpop dance <br>
   
   ![dance_gif](https://github.com/vuxminhan/Human-pose-feedback-system/assets/54212949/4b3e6c12-9f8b-4edd-be8d-a2014d1397b2)
   
3. **Martial Arts**: The application can analyze and provide visual feedback on posture, alignment, and movement precision crucial in martial arts. It helps in refining stances, kicks, and strikes through detailed pose analysis. <br>
   ### Taekwondon <br>

   ![taek_gif](https://github.com/vuxminhan/Human-pose-feedback-system/assets/54212949/73146a73-75cc-4ca4-9090-6b60926676a3)

   ### Vovinam <br>
   
   ![vovinam_gif (2)](https://github.com/vuxminhan/Human-pose-feedback-system/assets/54212949/881217ca-f8ee-4951-891c-db0a4601e990)

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

Please refer to [infer_wild.ipynb](infer_wild.ipynb).or [colab](https://colab.research.google.com/github/vuxminhan/Human-pose-feedback-system/blob/main/infer_wild.ipynb)



