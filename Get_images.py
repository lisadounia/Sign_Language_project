import os
import cv2


#creation of the directory 
data_dir="./data" #addresse
if not os.path.exists(data_dir): #create only if it doesn't exist
    os.makedirs(data_dir)#create

#class
alphabet=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
n_classes=len(alphabet)
#how many pics per class
dataset_size=50

#capture of the pics
cap=cv2.VideoCapture(0)
for letter in range(n_classes):
    if not os.path.exists( os.path.join(data_dir,alphabet[letter])):
        os.makedirs(os.path.join(data_dir,alphabet[letter])) #ex: "./data/A"
    print('Collecting data for letter{}'.format(alphabet[letter]))

    #Display the camera
    while True: 
        ret,frame=cap.read(0) #ret = bool : if the frame is successfully read , frame : frame cupture by the cap
        #print text on the video
        cv2.putText(frame,"if ready to capture, press enter", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame) # Show the frame in a window named 'frame'
        # if cv2.waitKey(1) == 13:  #enter (in ascii) to change letter
        #     print('start')
        #     break
        if cv2.waitKey(1) == ord('q'): # q to quit
            print('start')
            break
            # cap.release()
            # cv2.destroyAllWindows()
            # exit() 

    #Capture the pics and add them in the correct file 
    count=0
    while count< dataset_size :
            ret, frame = cap.read()
            cv2.imshow('frame',frame)
            print('pic taken')
            cv2.waitKey(25) # letting 25 milisec between each pic
            #save the image
            cv2.imwrite(os.path.join(data_dir,alphabet[letter],"{}.jpg".format(count)),frame) 
            count+=1

#always release the cam and close window
cap.release()
cv2.destroyAllWindows()


        
