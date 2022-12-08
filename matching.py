import cv2
import numpy as np
import os
from time import sleep
path='ImagesQuery'
orb=cv2.ORB_create(nfeatures=500)
images=[]
className=[]
myList=os.listdir(path)
print('Total Classes Detected',len(myList))
for cl in myList:
    imgCur = cv2.imread(f'{path}/{cl}',0)
    images.append(imgCur)
    className.append(os.path.splitext(cl)[0])
print(className)

def findDes(images):
    desList=[]
    for img in images:
        kp,des=orb.detectAndCompute(img,None)
        desList.append(des)
    return desList

def findID(img,desList,thres=15):
    kp2,des2=orb.detectAndCompute(img,None)
    bf=cv2.BFMatcher()
    matchList=[]
    finalVals=-1
    try:
        for des in desList:
            matches=bf.knnMatch(des,des2,k=2)
            good=[]
            for m,n in matches:
                if m.distance < 0.75 * n.distance:
                     good.append([m])
            matchList.append(len(good))
        #print(matchList)
    except:
        pass
    if len(matchList)!=0:
        if max(matchList)>thres:
            finalVals=matchList.index(max(matchList))
    return finalVals

desList=findDes(images)
print(desList)

cap=cv2.VideoCapture(0)
while True:
    success , img2=cap.read()
    imgOriginal=img2.copy()
    img2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    id=findID(img2,desList)
    if id!=-1:
        cv2.putText(imgOriginal,className[id],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)
    else:
        cv2.putText(imgOriginal,"Object doesn't exist in database", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    cv2.imshow('OBJECT DETECTION AND MATCHING',imgOriginal)
    cv2.waitKey(1)






# import cv2
# import numpy as np

# img1 = cv2.imread('ImagesQuery/PS4-Detroit.jpg',0)
# img2 = cv2.imread('Train/Train1.jpg',0)
#
# orb = cv2.ORB_create(nfeatures = 1000)
#
# kp1, des1 = orb.detectAndCompute(img1,None)
# kp2, des2 = orb.detectAndCompute(img2,None)
#
# # imgkp1 = cv2.drawKeypoints(img1,kp1,None)
# # imgkp2 = cv2.drawKeypoints(img2,kp2,None)
#
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1,des2,k=2)
#
# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])
#
# print(len(good))
# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
# imgKp1 = cv2.drawKeypoints(img1,kp1,None)
# imgKp2 = cv2.drawKeypoints(img2,kp2,None)
#
# # cv2.imshow('img1',img1)
# # cv2.imshow('img2',img2)
# # cv2.imshow('kp1',imgKp1)
# # cv2.imshow('kp2',imgKp2)
# cv2.namedWindow("output", cv2.WINDOW_NORMAL)
# imS = cv2.resize(img3, (1260, 940))
# cv2.imshow("output", imS)
#
# cv2.waitKey(0)