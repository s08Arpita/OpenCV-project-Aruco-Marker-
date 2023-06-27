# final working file

import cv2
import numpy as np

Colors = [
    [[32, 10, 20], [161, 255, 255]],  # Green
    [[14, 185, 185], [173, 255, 255]],  # Orange
    [[0, 0, 0], [179, 255, 184]],  # Black
    [[0, 0, 205], [179, 40, 233]]  # Peach-pink
]

src = []
dst = []


def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            if objCor == 4:
                aspRatio = w / float(h)
                if aspRatio > 0.95 and aspRatio < 1.05:
                    print(approx)
                    print("square")
                    dst = approx
                    return dst


img = cv2.imread("C:\\Users\\Arpita Singh\\PycharmProjects\\resource\\CVtask.jpg")
img = cv2.resize(img, (440, 310))
imgResult = img.copy()
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

arucos = [
    "C:\\Users\\Arpita Singh\\OneDrive\Desktop\\OpenCV-project-Aruco-Marker--main\\LMAO.jpg",
    "C:\\Users\\Arpita Singh\\OneDrive\Desktop\\OpenCV-project-Aruco-Marker--main\\XD.jpg",
    "C:\\Users\\Arpita Singh\\OneDrive\Desktop\\OpenCV-project-Aruco-Marker--main\\Ha.jpg",
    "C:\\Users\\Arpita Singh\\OneDrive\Desktop\\OpenCV-project-Aruco-Marker--main\\HaHa.jpg",
]


def detectAruco(img):
    ARUCO_DICT = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
        "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
        "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
        "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
    }

    for (arucoName, arucoDict) in ARUCO_DICT.items():
        arucoDict = cv2.aruco.Dictionary_get(arucoDict)
        arucoParams = cv2.aruco.DetectorParameters_create()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)
        if len(corners) > 0:
            print(f"[INFO] detected {len(corners)} markers for '{arucoName}'")


for aruco in arucos:
    imgAruco = cv2.imread(aruco)
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(imgAruco, arucoDict, parameters=arucoParams)
    arucoID = ids[0][0]

    imgB = np.zeros((imgAruco.shape[1], imgAruco.shape[0], 3), np.uint8)
    # print(imgBlack)
    cv2.drawContours(imgB, [corners[0].astype(int)], -1, (255, 255, 255), cv2.FILLED)

    imgBlack = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
    imgSrc = cv2.bitwise_and(imgAruco, imgB)

    contours, hierarchy = cv2.findContours(imgBlack, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for curve in contours:
        peri = cv2.arcLength(curve, True)
        approx = cv2.approxPolyDP(curve, 0.02 * peri, True)
        src = approx
        # for color detection
        lower = np.array(Colors[arucoID - 1][0])
        upper = np.array(Colors[arucoID - 1][1])
        mask = cv2.inRange(imgHSV, lower, upper)
        dst = getContours(mask)
        h, status = cv2.findHomography(src, dst)  # doing the maths
        imgTemp = cv2.warpPerspective(imgSrc, h, (440, 310))
        imgResult = cv2.bitwise_or(imgTemp, imgResult)

cv2.imshow("Result", imgResult)

cv2.waitKey(0)
