import cv2
import pickle
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./MLP_model.p','rb'))
model=model_dict['model']

cap=cv2.VideoCapture(0)

#detection of hand

mp_hands=mp.solutions.hands #Use hand detection
mp_drawing = mp.solutions.drawing_utils # draw lines on the image
mp_drawing_styles = mp.solutions.drawing_styles  #style of the landmarks and conection
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3) #extract landmarks, static ( it analyzes each image independently), need to be 30% sure to have detected a hand to return landmarks 


alphabet=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","space"]

word=""
while True : 
    ret,frame=cap.read()

    #position of the letters displayed 
    H, W, _ = frame.shape  
    x_text = W // 2 - 30  
    y_text = H - 20 


    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)#MediaPipe only use RGB color

    results = hands.process(frame_rgb) # detect hand and landmark on the image
    if results.multi_hand_landmarks : # if a hand is detected, there is all the landmarks of the hand in the object result
        for hand_landmarks in results.multi_hand_landmarks : #for each hand detected
            mp_drawing.draw_landmarks(
                frame,  
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,  
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())     
            # mp_drawing.draw_landmarks( frame, set of 21 landmarks, connection between the landmarks, landmark style of drawing , connection style of drawing )    

        for hand_landmarks in results.multi_hand_landmarks : #for each hand detected
                data_landmark=[]
                for i in range(len(hand_landmarks.landmark)): #each i is a list of 21 landmarks which represent a hand in a list
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    

                # there will be 21 elements in the list
                    data_landmark.append(x)
                    data_landmark.append(y)

        #start classify live
        #predicted_letter = model.predict([np.asarray(data_landmark)]) # convert the list of landmark into a nparray, and we put the array into bracket to make a 2D array ( we need a 2D array for prediction)
        
        #MLP
        predicted_letter_number = model.predict([np.asarray(data_landmark)])
        predicted_letter_number=predicted_letter_number[0]
        predicted_letter=alphabet[predicted_letter_number]
        # prediction output : an numpy array with the letter in it ex: ['A']

        #display the letter on the screen
        cv2.putText(frame, predicted_letter, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 5, cv2.LINE_AA)  # black contour
        cv2.putText(frame, predicted_letter, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2, cv2.LINE_AA)  # white text



    
    cv2.imshow('Sign Language Detection',frame)
    key = cv2.waitKey(25) #25 milisecs bewteen frames
    if key == ord('q') or key == 27:  # 'q' or 'ESC' (27 in ASCII)
        break
cap.release()
cv2.destroyAllWindows()
