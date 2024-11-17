from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os 
import csv
import time
from datetime import datetime
from win32com.client import Dispatch


def speak(str1):
    speak=Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if not os.path.exists('data/'):
    os.makedirs('data/')

with open('data/names.pk1', 'rb') as f:
    LABELS = pickle.load(f)

knn=KNeighborsClassifier(n_neighbors=5)

knn.fit(FACES, LABELS)
imgBackground=cv2.imread("bgimg.png")

COL_NAMES=['NAME', 'VOTE', 'DATE' , 'TIME']

while True:
    ret, frame = video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray, 1.3 ,5)
    for(x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img=cv2.resize(crop_img, (50,50)).flatten().reshape(1, -1)
        output=knn.predict(resized_img)
        ts=time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp=datetime.fromtimestamp(ts).strftime("%H-%M-%S")
        exist=os.path.isfile("Votes" + ".csv")
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x,y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (s, y-15), cv2.FONT_HERSHEY_COMPLEX, 1,  (50, 50, 255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50, 50, 255), 1)
        attendence=[output[0], timestamp]

    imgBackground[]    
    cv2.imshow('frame', imgBackground)    
    k=cv2.waitKey( 1)

    def check_iff_exists(value):
        try:
            with open("Vote.csv", "r") as csvfile:
                reader= csv.reader(csvfile)
                for row in reader:
                    if row and row[0]==value:
                        return True
                    
        except FileNotFoundError:
            print("file not found or unable to open the CSV file.")
        return False

    voter_exist = check_if_exists(output[0])
    if voter_exist:
        speak("you have already voted")
    if k==ord("1"):
        speak("your vote has been recorded")
        if exists:
            with open("votes" + ".csv", "+a") as csvfile:
                writer=csv.writer(csvfile)
                attendence = [output[0], "BJP", date, timestamp]
                writer.writerow(attendence)
            csvfile.close()
        speak("thank you for participating in the elections")    
        break
video.release()
cv2.destroyAllWindows()
