#!/usr/bin/env python2

"""
OpenCV example. Show webcam image and detect face.
"""

# Downscale the image supplied to opencv by this factor. Increase
# if the recognition is to slow.
DOWNSCALE = 2
# take a photo if a face is visible for longer than this amount of seconds
PHOTOTRESHOLD = 60
# your webcam's resolution
CAMRES = (800, 448)
# your screen's resolution
SCREENRES = (1366, 768)


import cv2
#import cv2.cv as cv
import time
import os
import subprocess

webcam = cv2.VideoCapture(0)
cv2.namedWindow("pyCamLock",cv2.WND_PROP_FULLSCREEN)
# TRAINSET
filepath = os.path.realpath(__file__)
folderpath = os.path.abspath(os.path.join(filepath, os.pardir))

# Location of CV face cascades on your system
cc_frontal=folderpath+"/opencvData/lbpcascades/lbpcascade_frontalface.xml"
#print (cc_frontal)

# directory where photos are stored
PHOTODIR = folderpath+"/images"

classifier = cv2.CascadeClassifier(cc_frontal)

viewFlag=True

#webcam.set(cv2.CV_CAP_PROP_FRAME_WIDTH, CAMRES[0])
#webcam.set(cv2.CV_CAP_PROP_FRAME_HEIGHT, CAMRES[1])

if webcam.isOpened():  # try to get the first frame
    rval, frame = webcam.read()
else:
    rval = False



if not os.path.isfile(cc_frontal):
    print("Face cascades not found. Face recognition disabled.")

lastDetected= time.time()
lastRec = 0
while rval:

    # detect faces and draw bounding boxes
    minisize = (frame.shape[1] / DOWNSCALE, frame.shape[0] / DOWNSCALE)
    miniframe = cv2.resize(frame, minisize)
    faces = classifier.detectMultiScale(miniframe)
    x, y = 100, 100
    detected = False
    for f in faces:
        x, y, w, h = [v * DOWNSCALE for v in f]                
        if w > 100:
            detected = True
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 6)
            #cv2.rectangle(miniframe, (x / DOWNSCALE, y / DOWNSCALE), ((x + w) / DOWNSCALE, (y + h) / DOWNSCALE), (0, 0, 255))
            lastDetected=time.time()




    if time.time() - lastDetected > 3:
        subprocess.call('/System/Library/CoreServices/Menu\ Extras/User.menu/Contents/Resources/CGSession -suspend', shell=True)

    if viewFlag or not detected:# live preview
        pframe = cv2.resize(frame, SCREENRES)
        cv2.putText(pframe, "testing", (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255))
        #cv2.setWindowProperty("pyCamLock",cv2.WND_PROP_FULLSCREEN,cv.CV_WINDOW_FULLSCREEN)
        cv2.imshow("pyCamLock", pframe)

    # record
    if detected and len(faces) > 0 and time.time() - lastRec > PHOTOTRESHOLD:
        if not os.path.exists(PHOTODIR):
            os.makedirs(PHOTODIR)
        cv2.imwrite("%s/%f.jpg" % (PHOTODIR,time.time()), miniframe)
        lastRec = time.time()

    # get next frame
    rval, frame = webcam.read()

    key = cv2.waitKey(20)
    if key in [27, ord('Q'), ord('q')]:  # exit on ESC
        break
