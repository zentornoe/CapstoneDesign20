import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import mahotas as mh
from sklearn.preprocessing import StandardScaler

def image_bite_regulate(image_glob):
    for i in image_glob:
        im=cv2.imread(i)
        im=cv2.resize(im,(512,512))
        cv2.imwrite(i,im)


def _crop(file_path):  # INPUT : file path that want to open
    maxArea = 0
    maxNum = 0
    # angle = 0.0

    img = file_path
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thr = cv2.threshold(gray, 127, 255, 0)
    cnt, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(cnt)):  # calculate contours which has maximum area
        area = cv2.contourArea(cnt[i])
        if maxArea < area:
            maxArea = area
            maxNum = i

    x,y,w,h = cv2.boundingRect(cnt[maxNum])  # Calculate minimum area rectangle
    roi = img[y:y+h, x:x+w]

    return roi

def _crop_expand(file_path):     # INPUT : file path that want to open
    img = file_path     # read image

    h, w, c = img.shape             # get height, width, channel of image

    nh = int(h*1.2)                 # root(2) = 1.41 ... -> resizing ratio = 1.5
    nw = int(h*1.2)

    blank_img = np.zeros((nh, nw, c), np.uint8) # generate blank resized image
    blank_img[:, :] = (0, 0, 0)

    #calculate offset to center image
    result = blank_img.copy()
    x_offset = int((nh-h)/2)
    y_offset = int((nw-w)/2)

    result[0:+h, 0:+w] = img.copy()   # overwrite the orignial image to blank image
    result=cv2.resize(result,(512,512))

    return result  # open_cv image

def _angle(cv_image):  # INPUT : cv2 image (cv2.imread(file_path))
    maxArea = 0
    maxNum = 0

    img = cv_image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thr = cv2.threshold(gray, 127, 255, 0)
    cnt, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(cnt)):  # calculate contours which has maximum area
        area = cv2.contourArea(cnt[i])
        if maxArea < area:
            maxArea = area
            maxNum = i

    rect = cv2.minAreaRect(cnt[maxNum])  # Calculate minimum area rectangle
    box = cv2.boxPoints(rect)

    # Rotated angle
    if ((pow(box[0][0] - box[1][0], 2) + pow(box[0][1] - box[1][1], 2)) >= (
            pow(box[1][0] - box[2][0], 2) + pow(box[1][1] - box[2][1], 2))):
        return -90 + rect[2]
    else:
        return rect[2]


# expand image w/o resizing
def _expand(cv_image):  # INPUT : cv2 image (cv2.imread(file_path))
    img = cv_image  # read image
    h, w, c = img.shape  # get height, width, channel of image
    # root(2) = 1.41 ... -> resizing ratio = 1.5
    nh = int(h * 1.5)
    nw = int(w * 1.5)

    blank_img = np.zeros((nh, nw, c), np.uint8)  # generate blank resized image
    blank_img[:, :] = (0, 0, 0)

    # calculate offset to center image
    result = blank_img.copy()
    x_offset = int((nh - h) / 2)
    y_offset = int((nw - w) / 2)

    # overwrite the orignial image to blank image
    result[y_offset:y_offset + h, x_offset:x_offset + w] = img.copy()

    return result  # open_cv image


# rotate image
def _rotate(cv_image, angle):  # INPUT : cv2 image (cv2.imread(file_path))
    img = cv_image  # read image
    h, w, c = img.shape  # Get the height, width, channel of image

    # Set the rotation axis, angle, and scale
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), 90 + angle, 1)
    # result(rotated image) : img -> (angle) rotated
    rotated = cv2.warpAffine(img, mat, (w, h))

    return rotated  # cv2 image


# Area division & filling
def _division(cv_image):  # INPUT : cv2 image (cv2.imread(file_path))
    maxArea = 0
    maxNum = 0
    maxNum2 = 0

    img = cv_image  # read image
    # convert color file to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thr = cv2.threshold(gray, 127, 255, 0)  # get threshold
    cnt, _ = cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # find contours

    for i in range(len(cnt)):  # calculate contours which has maximum area
        area = cv2.contourArea(cnt[i])
        if maxArea < area:
            maxArea = area
            maxNum2 = maxNum
            maxNum = i

    cv2.fillPoly(img, pts=[cnt[maxNum2]], color=(0, 0, 0))  # 2nd Contour filling

    return img


# Fractured bone w/o overlap
def _separate(cv_image):
    maxArea = 0
    maxArea2 = 0
    maxNum = 0
    maxNum2 = 0

    img = cv_image  # read image
    img1 = np.copy(img)  # copied image
    # convert color file to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thr = cv2.threshold(gray, 127, 255, 0)  # get threshold
    cnt, _ = cv2.findContours(
        thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # find contours

    for i in range(len(cnt)):  # calculate contours which has maximum area
        area = cv2.contourArea(cnt[i])
        if maxArea2 < area:
            if maxArea < area:
                maxNum2 = maxNum
                maxNum = i
                maxArea2 = maxArea
                maxArea = area
            else:
                maxNum2 = i
                maxArea2 = area

    ct0 = cnt[maxNum]
    ct2 = cnt[maxNum2]
    h, _, _ = img.shape
    max0 = h
    max2 = h

    for j in range(len(ct0)):
        if max0 > ct0[j][0][1]:
            max0 = ct0[j][0][1]

    for k in range(len(ct2)):
        if max2 > ct2[k][0][1]:
            max2 = ct2[k][0][1]

    if (max0 < max2):
        cv2.fillPoly(img, pts=[cnt[maxNum2]], color=(0, 0, 0))  # 2nd Contour filling
        cv2.drawContours(img, [cnt[maxNum2]], 0, (0, 0, 0), 5)
        cv2.fillPoly(img1, pts=[cnt[maxNum]], color=(0, 0, 0))  # 1st Contour filling
        cv2.drawContours(img1, [cnt[maxNum]], 0, (0, 0, 0), 5)
    else:
        cv2.fillPoly(img1, pts=[cnt[maxNum2]], color=(0, 0, 0))  # 2nd Contour filling
        cv2.drawContours(img1, [cnt[maxNum2]], 0, (0, 0, 0), 5)
        cv2.fillPoly(img, pts=[cnt[maxNum]], color=(0, 0, 0))  # 1st Contour filling
        cv2.drawContours(img, [cnt[maxNum]], 0, (0, 0, 0), 5)

    return img, img1  # upper, lower


# Fractured bone with overlap
def _fracWhere(cv_image):  # INPUT : cv2 image
    maxArea = 0
    maxNum = 0
    clst = 0
    clst_ = 0
    clst1 = 0
    clst2 = 0

    img = cv_image  # read image
    img1 = np.copy(img)
    # convert color file to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thr = cv2.threshold(gray, 127, 255, 0)  # get threshold
    cnt, _ = cv2.findContours(
        thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # find contours
    h, w, _ = img.shape
    center = int(h / 2)
    clst = h
    clst_ = h
    print('Center : ' + str(center))

    for i in range(len(cnt)):  # calculate contours which has maximum area
        area = cv2.contourArea(cnt[i])
        if maxArea < area:
            maxArea = area
            maxNum = i
    ct = cnt[maxNum]

    cv2.drawContours(img, [cnt[maxNum]], 0, (125, 125, 0), 3)

    hull = cv2.convexHull(cnt[maxNum], returnPoints=False)
    defects = cv2.convexityDefects(cnt[maxNum], hull)

    for i in range(defects.shape[0]):
        sp, ep, fp, dist = defects[i, 0]
        start = tuple(ct[sp][0])
        end = tuple(ct[ep][0])
        farthest = tuple(ct[fp][0])

        if (clst_ > abs(farthest[1] - center)):
            if (clst > abs(farthest[1] - center)):
                clst_ = clst
                clst = abs(farthest[1] - center)
                clst2 = clst1
                clst1 = farthest[1]
            else:
                clst_ = abs(farthest[1] - center)
                clst2 = farthest[1]

    return clst1, clst2


# Separate overlapped images
def _cutting(cv_image, pt1, pt2):
    h, w, _ = cv_image.shape  # height, width, channel

    img0 = cv_image
    img1 = np.copy(img0)

    if pt2 < pt1:
        fst = pt1
        snd = pt2
    else:
        fst = pt2
        snd = pt1

    cv2.rectangle(img0, (0, snd), (w, h), (0, 0, 0), -1)  # Fill the lower part
    cv2.rectangle(img1, (0, 0), (w, fst), (0, 0, 0), -1)  # Fill the upper part

    return img0, img1

def image_similarity(image_01, ch):

    images = glob('stl_dataset/*.png')
    image_bite_regulate(images)


    #사진들 측징 및 특성 추출
    features = []

    for im in images:
        im = mh.imread(im)
        features.append(mh.features.haralick(im).ravel())

    features[0]=mh.features.haralick(image_01).ravel()

    features = np.array(features)
    sc = StandardScaler()
    features = sc.fit_transform(features)
    from scipy.spatial import distance
    dists = distance.squareform(distance.pdist(features))

    return plotImages(0,image_01,dists,images,ch)


#각각의 시진들 유사도에 따라 배열
def selectImage(n, m, dists, images):
    image_position = dists[n].argsort()[m]
    image = mh.imread(images[image_position])
    return image


def plotImages(n,image_01,dists,images,ch):
    angle=[]

    if(ch==1):
        plt.figure(figsize=(10, 5))
        plt.subplot(1,3,1)
        plt.imshow(image_01)
        plt.title('Original')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1,3,2)
        plt.imshow(selectImage(n, 1, dists, images))
        plt.title('1st Similar Degree:'+str(dists[n].argsort()[1]*5))
        angle.append(dists[n].argsort()[1] * 5)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1, 3, 3)
        plt.imshow(selectImage(n, 2, dists, images))
        plt.title('2nd Similar Degree:' + str(dists[n].argsort()[2] * 5))
        angle.append(dists[n].argsort()[2] * 5)
        plt.xticks([])
        plt.yticks([])


        plt.show()
        return angle

    if (ch==2):
        angle.append(dists[n].argsort()[1] * 5)
        angle.append(dists[n].argsort()[2] * 5)
        return angle




def plotshow2(image1,image2,ang0,ang1):
    plt.figure(figsize=(10, 5))
    image=glob('stl_dataset/*.png')

    plt.subplot(2, 3, 1)
    plt.imshow(image1)
    plt.title('Upper Bone')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 3, 4)
    plt.imshow(image2)
    plt.title('Under Bone')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 3, 2)
    mh.imread(image[int(ang0[0]/5)])
    plt.imshow(mh.imread(image[int(ang0[0]/5)]))
    plt.title('1st Similar Degree:' + str(ang0[0]))
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 3, 3)
    a1 = int(ang0[1]/5)
    mh.imread(image[a1])
    plt.imshow(mh.imread(image[a1]))
    plt.title('2st Similar Degree:' + str(ang0[1]))
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 3, 5)
    a1 = int(ang1[0] / 5)
    mh.imread(image[a1])

    plt.imshow(mh.imread(image[a1]))
    plt.title('1st Similar Degree:' + str(ang1[0]))
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 3, 6)
    a1=int(ang1[1]/5)
    mh.imread(image[a1])
    plt.imshow(mh.imread(image[a1]))
    plt.title('2st Similar Degree:' + str(ang1[1]))
    plt.xticks([])
    plt.yticks([])


    plt.show()
