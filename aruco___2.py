
# final aurco reader

import cv2
import numpy as np
import imutils

center = []

id_no=[]
#store all addr.
colour = []
arco_id = ["C:\\Users\\Arpita Singh\\PycharmProjects\\resource\\LMAO.jpg",
	"C:\\Users\\Arpita Singh\\PycharmProjects\\resource\\XD.jpg",
	"C:\\Users\\Arpita Singh\\PycharmProjects\\resource\\Ha.jpg",
	"C:\\Users\\Arpita Singh\\PycharmProjects\\resource\\HaHa.jpg"]
cv_task = "C:\\Users\\Arpita Singh\\PycharmProjects\\resource\\CVtask.jpg"
Colors = [
[[[79,209,146]]], # Green
[[[9,127,240]]], # Orange
[[[0, 0, 0]]], # Black
[[[210,222,228]]], # Peach-pink
]



# find the id of each arco

#function reading ids of marker
def findAurco(img,marker_size=5,total_marker=250,draw=True):
    grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    key=getattr(cv2.aruco,f'DICT_{marker_size}X{marker_size}_{total_marker}')
    arucoDict=cv2.aruco.Dictionary_get(key)
    arucoParem=cv2.aruco.DetectorParameters_create()
    (bbox,ids,_)=cv2.aruco.detectMarkers(grey,arucoDict,parameters=arucoParem)
    arucoID = ids[0][0]
    print(arucoID)
    id_no.append(arucoID)



for id in arco_id:
    marker_name = cv2.imread(id)
    print('ID OF ',id)
    findAurco(marker_name)

# rename all the image name with their id
i=0
for id in arco_id:
    marker_name = cv2.imread(id)
    num = '{}.jpg'.format(id_no[i])
    #cv2.imshow(num,marker_name)
    i=i+1
    cv2.imwrite(num, marker_name)

# make contour on image
img = cv2.imread(cv_task)
img = cv2.resize(img,None,None,fx=0.5,fy=0.5)
#cv2.imshow('original',img)
# covert into the gray image
img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',img_grey)
#threshold image
thresh, thresh_img = cv2.threshold(img_grey,225,255,cv2.THRESH_BINARY)
#cv2.imshow('thresh',thresh_img)
#find the total contour
cont = cv2.findContours(thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cont = imutils.grab_contours(cont)
print(len(cont))
for c in cont:
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.01 * perimeter, True)
    x, y, w, h = cv2.boundingRect(approx)
    if len(approx) == 4:
        aspRatio = w / float(h)
        if aspRatio > 0.95 and aspRatio < 1.05:

            box = cv2.minAreaRect(c)
            print(box[0])
            center.append(box[0])

            (y, x) = box[0]
            x= int(x)
            y= int(y)
            color = img[x:x+1,y:y+1]
            lis = color.tolist()
            a=0
            for c in Colors:
                a= a+1
                if lis == c:
                    print(a)
                    break

            box = cv2.boxPoints(box)
            print(box)
            box = np.array(box, dtype='int')
            cv2.drawContours(img, [box], -1, (0, 255, 0),3)


            marker = cv2.imread(arco_id[a-1])
            key = getattr(cv2.aruco, f'DICT_{5}X{5}_{250}')
            arucoDict = cv2.aruco.Dictionary_get(key)
            arucoParem = cv2.aruco.DetectorParameters_create()
            (bbox, ids, _) = cv2.aruco.detectMarkers(marker, arucoDict, parameters=arucoParem)
            bbox = np.array(bbox, dtype='int')
            bbox=bbox[0]
            #h, w, c = marker.shape
            pts1 = np.array(box)
            pts2 = np.array(bbox)
            matrix, _ = cv2.findHomography(pts2, pts1)
            imgOut = cv2.warpPerspective(marker, matrix, (img.shape[1], img.shape[0]))
            img = cv2.bitwise_or(img, imgOut)    # 0 for black and 1 for colour

cv2.imshow('out',img)
cv2.imwrite('finally.jpg',img)
#cv2.imshow('imgOut',imgOut)



cv2.waitKey(0)
