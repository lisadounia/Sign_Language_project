#this code will take the landmark of the pictures to help us be the reference for the classifier
import pickle
import os 
import mediapipe as mp
import cv2

#detection of hand
mp_hands=mp.solutions.hands #Use hand detection
mp_drawing = mp.solutions.drawing_utils # draw dts, lines on the image
    #mp_drawing_styles = mp.solutions.drawing_styles 
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3) #extract landmarks, static ( it analyzes each image independently), need to be 30% sure to have detected a hand to return landmarks 



data_dir="./data"
data=[] # coordinate of the landmarks
labels=[] #name of the letter 

for dir_ in os.listdir(data_dir): #each file
    dir_path = os.path.join(data_dir, dir_)  
    if os.path.isdir(dir_path):  # check if it is a directory
        print(dir_)
        for img_path in os.listdir(dir_path): #each image in the file

            img=cv2.imread(os.path.join(dir_path, img_path)) #load the image
            img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)#MediaPipe only use RGB color

            results = hands.process(img_rgb) # detect hand and landmark on the image
            if results.multi_hand_landmarks : # if a hand is detected, there is all the landmarks of the hand in the object result
                for hand_landmarks in results.multi_hand_landmarks : #for each hand detected
                    data_landmark = []
                    for i in range(len(hand_landmarks.landmark)): #each i is a landmark 
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                    # there will be 21 elements in the list
                        data_landmark.append(x)
                        data_landmark.append(y)
                    data.append(data_landmark) # for each and we will have all the 21 landmarks x,y
                    labels.append(dir_) # dir_ named after the letters


data_dict ={'data': data, 'labels': labels}
f = open('data.p', 'wb')  # wb : write in binary 
pickle.dump(data_dict, f) #model contain all the parameters learn 
f.close()

