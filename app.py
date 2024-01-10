
from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from sklearn.metrics import accuracy_score


from win32com.client import Dispatch
def speak(str1):
    speak=Dispatch(("SAPI.SpVoice"))
    speak.speak(str1)
# import pyttsx3

# from datetime import datetime


# engine = pyttsx3.init()
# """VOICE"""
# voices = engine.getProperty('voices')       #getting details of current voice
# #engine.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male
# engine.setProperty('voice', voices[1].id) 

# with open('data/faces_data.pkl', 'rb') as f:
#     array = pickle.load(f)

video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

with open('data/names.pkl', 'rb') as w:
    LABELS=pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES=pickle.load(f)


knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

split_ratio=0.8
split_index=int(len(FACES)* split_ratio)

train_faces, test_faces= FACES[:split_index], FACES[split_index:]
train_labels, test_labels= LABELS[:split_index], LABELS[split_index:]

knn.fit(train_faces, train_labels)

predictions= knn.predict(test_faces)

accuracy= accuracy_score(test_labels, predictions)
print(f"Accuracy:{accuracy*100:.2f}%")

while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray, 1.3 ,5)
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w, :]
        resized_img=cv2.resize(crop_img, (50,50)).flatten().reshape(1,-1)
        output=knn.predict(resized_img)
        
        
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        # cv2.rectangle(frame,(0,0),(1000,45),(50,50,255),-1)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        cv2.putText(frame," Welcome To Virinchi College.", (10,35), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
        cv2.putText(frame, str(output[0]), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
        speak("Welcome to virinchi college"+str(output[0]))
        #cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
        # engine.say(str(output[0]))
        
         # Save the face of the unknown person to the training data
        # FACES = np.vstack([FACES, resized_img])
        # LABELS = np.append(LABELS, 'Unknown')
        
    frame = cv2.resize(frame,(1500,1000))
    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()

# adding unknown face 
# from sklearn.neighbors import KNeighborsClassifier
# import cv2
# import pickle
# import numpy as np
# import os
# from win32com.client import Dispatch

# def speak(str1):
#     speak=Dispatch(("SAPI.SpVoice"))
#     speak.speak(str1)

# video = cv2.VideoCapture(0)
# facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# with open('data/names.pkl', 'rb') as w:
#     LABELS = pickle.load(w)
# with open('data/faces_data.pkl', 'rb') as f:
#     FACES = pickle.load(f)

# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(FACES, LABELS)

# split_ratio = 0.8
# split_index = int(len(FACES) * split_ratio)

# train_faces, test_faces = FACES[:split_index], FACES[split_index:]
# train_labels, test_labels = LABELS[:split_index], LABELS[split_index:]

# knn.fit(train_faces, train_labels)

# predictions = knn.predict(test_faces)

# accuracy = accuracy_score(test_labels, predictions)
# print(f"Accuracy: {accuracy * 100:.2f}%")

# while True:
#     ret, frame = video.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
#     for (x, y, w, h) in faces:
#         crop_img = frame[y:y+h, x:x+w, :]
#         resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
#         output = knn.predict(resized_img)
        
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
#         cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
#         cv2.putText(frame, " Welcome To Virinchi College.", (10, 35), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
#         cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
#         speak("Welcome to Virinchi College" + str(output[0]))
        
#         # Save the face of the unknown person to the training data
#         FACES = np.vstack([FACES, resized_img])
#         LABELS = np.append(LABELS, 'Unknown')
        
#     frame = cv2.resize(frame, (1500, 1000))
#     cv2.imshow("Frame", frame)
    
#     k = cv2.waitKey(1)
#     if k == ord('q'):
#         break

# # Save the updated training data
# with open('data/faces_data.pkl', 'wb') as f:
#     pickle.dump(FACES, f)
# with open('data/names.pkl', 'wb') as w:
#     pickle.dump(LABELS, w)

# video.release()
# cv2.destroyAllWindows()
