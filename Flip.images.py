import cv2
import os
import random

data_dir = "./data"

# Parcourir chaque dossier dans ./data
for dir_ in os.listdir(data_dir):
    dir_path = os.path.join(data_dir, dir_)
    
    # Vérifier que c'est bien un dossier
    if os.path.isdir(dir_path):
        print(dir_path)

        # Récupérer toutes les images dans le dossier
        images = []
        for img_file in os.listdir(dir_path):
            if img_file.lower().endswith(".jpg") or img_file.lower().endswith(".png"):
                images.append(img_file)

        num_images = len(images)

        # Vérifier qu'il y a des images dans le dossier
        if num_images > 0:

            # Déterminer aléatoirement combien d'images retourner (entre 25% et 20%)
            print(num_images)
            num_to_flip = random.randint(num_images // 5, num_images // 4)

            # Sélectionner aléatoirement les images à retourner
            images_to_flip = random.sample(images, num_to_flip)

            # Appliquer l'effet miroir aux images sélectionnées
            for image_file in images_to_flip:
                image_path = os.path.join(dir_path, image_file)
                img = cv2.imread(image_path)

                if img is not None:
                    # Appliquer le flip horizontal
                    img_flipped = cv2.flip(img, 1)
                    
                    # Sauvegarder en écrasant l'image originale
                    cv2.imwrite(image_path, img_flipped)

            print(f"{num_to_flip}/{num_images} images retournées dans : {dir_path}")
