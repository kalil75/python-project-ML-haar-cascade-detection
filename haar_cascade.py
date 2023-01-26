

import cv2 as cv
import numpy as np

im= cv.imread('mor_teams5.jpg')


gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Image', gray)

haar_cascade_modele = cv.CascadeClassifier('haar_face.xml')

faces_r = haar_cascade_modele.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=11)

print(f'Number of faces found = {len(faces_r)}')

for (x,y,w,h) in faces_r:
    cv.rectangle(im, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected ', im)



cv.waitKey(0)

