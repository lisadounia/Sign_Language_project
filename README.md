
# **ASL Alphabet Detector**

This project detects and recognizes **American Sign Language (ASL) alphabet** using computer vision and machine learning. It leverages **Mediapipe for hand tracking** and an **MLP model trained on the ASL dataset**.

**By Lisa Dounia**

## **Dataset**  
[ASL Alphabet Dataset](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset)

## **Project Structure**  
```
Sign_Language_project/
│── alphabetASL_detector.py     # Main script for real-time detection
│── img2landmarks.py            # Extracts landmarks from images (database)
│── training_MLP.py              # Trains the MLP model with the landmarks
│── mlp_model.p                  # Trained MLP model
│── Flip_images.py               # Data augmentation (Flips some images of the dataset to create a mirror effect)
