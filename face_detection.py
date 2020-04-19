import cv2
import os
import shutil


# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print('Make sure that lighting conditons are sufficient')
name=input('Enter first name: ')
try: 
      
    # creating a folder named data 
    if not os.path.exists('data/train/'+name): 
        os.makedirs('data/train/'+name)
        os.makedirs('data/val/'+name)
    else:
        shutil.rmtree('data/train/'+name+'/')
        shutil.rmtree('data/val/'+name+'/')
        os.makedirs('data/train/'+name)
        os.makedirs('data/val/'+name)
# if not created then raise error 
except OSError: 
    print ('Error: Creating directory of data') 
# To capture video from webcam. 

cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')
i=0
while True:
    # Read the frame
    ret, img = cap.read()
    if(ret==False):
        continue
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.2, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)
        roi_color = img[y:y + h, x:x + w]
        if(i>80):
            cv2.imwrite('data/val/'+name+'/'+str(i)+'_face.jpg', roi_color)
        else:
            cv2.imwrite('data/train/'+name+'/'+str(i)+'_face.jpg', roi_color)
        i=i+1
    # Display
    cv2.imshow('Face Detection', img)
    if(i%10==0 and i!=0):
        print('Captured '+str(i)+' images of '+name)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27 or i==100:
        break
# Release the VideoCapture object
print('Done.')
cap.release()
cv2.destroyAllWindows()