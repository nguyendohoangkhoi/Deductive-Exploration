import cv2
import os
import numpy as np
def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop top
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop top
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop top
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame

mainForder = 'SPACE/data/Atari/MontezumaRevenge-v0'
#myForders = os.listdir(mainForder)
myForders = os.listdir(mainForder)
for forder in myForders:
    path = mainForder+"/"+"temp"
    images = []
    mylist = os.listdir(path)
    print("total images detected : {}".format(len(mylist)))
    for imgN in mylist:
        curImg = cv2.imread("{}/{}".format(path,imgN))
        curImg = cv2.resize(curImg, (160,160))
        #curImg = cv2.resize(curImg,(0,0),None,0.2,0.2)
        images.append(curImg)
        print(curImg.shape)
    stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
    stitcher.setPanoConfidenceThresh(0.0)
    print(images)
    (status, result) =  stitcher.stitch(images)
    if (status == cv2.STITCHER_OK):
        print("Generated Paranoma")

        cv2.imshow("temp", result)
        cv2.waitKey(0)
    else :
        print("Failed",status)
        cv2.imshow("temp",result)
        cv2.waitKey(10)
    break
