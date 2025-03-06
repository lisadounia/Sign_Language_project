#Some images will have a mirror effect to train the model not to become accustomed to a single direction/hand. 
import cv2
import os
import random

data_dir = "./data"

for dir_ in os.listdir(data_dir):
    dir_path = os.path.join(data_dir, dir_)
    if os.path.isdir(dir_path):
        print(dir_path)
        images = []
        for img_file in os.listdir(dir_path):
            if img_file.lower().endswith(".jpg") or img_file.lower().endswith(".png"):
                images.append(img_file)

        num_images = len(images)
        if num_images > 0:

            # bewteen 20% and 25% of the pics will be flipped)
            print(num_images)
            num_to_flip = random.randint(num_images // 5, num_images // 4)

            images_to_flip = random.sample(images, num_to_flip)

            for image_file in images_to_flip:
                image_path = os.path.join(dir_path, image_file)
                img = cv2.imread(image_path)

                if img is not None:
                    img_flipped = cv2.flip(img, 1)
                    
                    cv2.imwrite(image_path, img_flipped)
